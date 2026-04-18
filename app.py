import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data, suggest_target_column, generate_agent_narrative
from src.report import generate_pdf_report
from src.eda import basic_profile

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="DataAgentX",
    layout="wide",
    page_icon="🚀"
)

api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

# ======================================================
# PREMIUM CSS
# ======================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0d14; }
[data-testid="stSidebar"] { background: #0f1320; border-right: 1px solid #1e2535; }
[data-testid="stMetric"] { background: #141826; border: 1px solid #1e2535; border-radius: 12px; padding: 16px !important; }
.stButton > button { border-radius: 10px; font-weight: 600; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE — initialize all keys once
# ======================================================
DEFAULTS = {
    "df": None, "profile": None, "X": None, "y": None,
    "model": None, "problem_type": None, "target_col": None,
    "business_insights": None, "analyzed": False,
    "chat_history": [], "model_card": None,
    "agent_narrative": None, "shap_top_features": None,
    "cv_score": None, "residual_std": None,
    "feature_schema": None, "trigger_narrative": False
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# HELPERS
# ======================================================
def executive_summary(df, target):
    vals = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(vals) < 10:
        return {}
    n = max(1, int(len(vals) * 0.2))
    recent = vals.tail(n).mean()
    prev   = vals.head(len(vals) - n).mean()
    change = (recent - prev) / (abs(prev) + 1e-6)
    slope  = np.polyfit(range(len(vals)), vals, 1)[0]
    return {
        "trend":  "📈 Growing"   if slope > 0 else "📉 Declining",
        "change": change,
        "risk":   "🔴 High" if change < -0.1 else "🟡 Medium" if change < -0.05 else "🟢 Low",
        "avg":    vals.mean(),
        "max":    vals.max(),
        "min":    vals.min()
    }

def smart_alerts(df, target):
    vals = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(vals) < 20:
        return []
    n = max(1, int(len(vals) * 0.2))
    change = (vals.tail(n).mean() - vals.head(len(vals)-n).mean()) / (abs(vals.head(len(vals)-n).mean()) + 1e-6)
    alerts = []
    if change < -0.1:
        alerts.append({"type": "warning",
            "msg": f"⚠️ {target} dropped {abs(change)*100:.1f}% vs historical",
            "action": "Review pricing, increase promotions, check supply chain."})
    elif change > 0.1:
        alerts.append({"type": "success",
            "msg": f"🚀 {target} grew {abs(change)*100:.1f}% vs historical",
            "action": "Scale inventory, increase ad spend, expand markets."})
    z = np.abs((vals - vals.mean()) / (vals.std() + 1e-6))
    if (z > 3).mean() > 0.02:
        alerts.append({"type": "warning",
            "msg": f"⚠️ {(z>3).mean()*100:.1f}% outliers in {target}",
            "action": "Investigate data quality or exceptional events."})
    return alerts

def check_leakage(X, y, threshold=0.95):
    """Detect data leakage — correlation too high = suspicious"""
    leaky = []
    for col in X.columns:
        try:
            corr = abs(pd.Series(X[col].values).corr(pd.Series(y.values)))
            if corr > threshold:
                leaky.append((col, round(corr, 4)))
        except:
            pass
    return leaky

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown("## 🚀 DataAgentX")
st.sidebar.caption("AI Business Intelligence Platform")

with st.sidebar.expander("🔑 AI Status"):
    if api_key:
        st.success("✅ AI Connected")
    else:
        st.warning("⚠️ Add OPENAI_API_KEY to secrets")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Data Source")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if st.sidebar.button("🎯 Walmart Demo"):
    try:
        demo_df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
        st.session_state.update({**DEFAULTS,
            "df": demo_df,
            "profile": basic_profile(demo_df)
        })
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Demo failed: {e}")

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        st.session_state.update({**DEFAULTS,
            "df": raw,
            "profile": basic_profile(raw)
        })
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

df      = st.session_state.df
profile = st.session_state.profile

if df is not None:
    st.sidebar.success("✅ Dataset Ready")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Rows",    df.shape[0])
    c2.metric("Cols",    df.shape[1])
    miss = df.isnull().sum().sum()
    if miss > 0:
        st.sidebar.warning(f"⚠️ {miss} missing values")
    else:
        st.sidebar.success("✅ No missing values")

# ======================================================
# MAIN
# ======================================================
st.markdown("# 🚀 DataAgentX")
st.caption("Upload data → AI analyzes → Get business decisions")
st.divider()

if df is None:
    st.info("👈 Upload a CSV or use Walmart Demo from sidebar")
    st.stop()

# ======================================================
# TARGET SELECTION
# ======================================================
st.markdown("### 🎯 What do you want to predict?")

priority = ["Weekly_Sales","Sales","Revenue","Profit","Price","Target","Churn"]

# Get all numeric columns — no aggressive filtering
numeric_cols = []
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_cols.append(c)

if not numeric_cols:
    st.error("No numeric columns found in dataset!")
    st.stop()

# Smart default
if st.session_state.target_col not in numeric_cols:
    st.session_state.target_col = next(
        (c for c in priority if c in numeric_cols), numeric_cols[0]
    )

target = st.selectbox(
    "Select Target Column",
    numeric_cols,
    index=numeric_cols.index(st.session_state.target_col)
)

# If target changes — reset analysis
if target != st.session_state.target_col:
    st.session_state.analyzed    = False
    st.session_state.agent_narrative = None
    st.session_state.model_card  = None
st.session_state.target_col = target

# ======================================================
# ANALYZE BUTTON
# ======================================================
if st.button("🚀 Analyze", use_container_width=False):

    with st.spinner("Training models... this takes 2-3 minutes"):
        try:
            # ── Pipeline ──────────────────────────────
            X, y = prepare_features(df, profile, target, training=True)

            if X.shape[1] == 0:
                st.error("No features after processing!")
                st.stop()

            # Ensure purely numeric
            X = X.select_dtypes(include=[np.number]).fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            y = y.reset_index(drop=True)

            # Align index
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len].reset_index(drop=True)
            y = y.iloc[:min_len].reset_index(drop=True)

            # ── Leakage check ─────────────────────────
            leaky = check_leakage(X, y, threshold=0.95)
            if leaky:
                st.warning(f"⚠️ Potential data leakage detected:")
                for col, corr in leaky:
                    st.write(f"  • {col} → correlation {corr}")
                # Remove leaky features
                X.drop(columns=[c for c, _ in leaky], errors="ignore", inplace=True)

            # ── Train ─────────────────────────────────
            problem = detect_problem_type(y)
            results, best_model_name = train_models(X, y, problem)
            model = joblib.load("models/best_model.pkl")

            # ── Cross Validation ──────────────────────
            if problem == "regression":
                scores = cross_val_score(
                    model, X, y,
                    cv=KFold(5, shuffle=True, random_state=42),
                    scoring="r2"
                )
                cv_label = "R²"
            else:
                scores = cross_val_score(
                    model, X, y,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring="accuracy"
                )
                cv_label = "Accuracy"

            cv_mean = round(float(scores.mean()), 4)
            cv_std  = round(float(scores.std()), 4)

            residual_std = None
            if problem == "regression":
                residual_std = float(np.std(y - model.predict(X)))

            # ── SHAP ──────────────────────────────────
            shap_top = []
            insights = []
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
                mean_abs = np.abs(sv).mean(axis=0)
                top_idx  = np.argsort(mean_abs)[::-1][:5]
                shap_top = [(X.columns[i], round(float(mean_abs[i]), 4)) for i in top_idx]
                insights = generate_business_impact(sv, X, problem, target)
            except Exception as se:
                insights = [f"SHAP skipped: {se}"]

            # ── Save to session ───────────────────────
            model_card = {
                "model":    best_model_name,
                "problem":  problem,
                "rows":     df.shape[0],
                "features": X.shape[1],
                "target":   target,
                "performance": {
                    f"{cv_label} (CV)": cv_mean,
                    "CV Std": cv_std
                }
            }

            st.session_state.update({
                "X": X, "y": y, "model": model,
                "problem_type":     problem,
                "target_col":       target,
                "business_insights": insights,
                "analyzed":         True,
                "model_card":       model_card,
                "shap_top_features": shap_top,
                "cv_score":         cv_mean,
                "residual_std":     residual_std,
                "feature_schema":   X.columns.tolist(),
                "agent_narrative":  None,   # reset — will auto-generate below
            })

            st.success(f"✅ {best_model_name} trained! {cv_label}: {cv_mean:.4f}")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)   # shows full traceback in UI for debugging

# ======================================================
# RESULTS — always rendered from session state
# ======================================================
if st.session_state.analyzed:

    mc   = st.session_state.model_card or {}
    perf = mc.get("performance", {})

    st.divider()
    st.markdown("### 🏆 Model Performance")
    cols_perf = st.columns(4)
    cols_perf[0].metric("Best Model", mc.get("model", "—"))
    for i, (k, v) in enumerate(perf.items()):
        cols_perf[i+1].metric(k, f"{v:.4f}")
    cols_perf[3].metric("Features", mc.get("features", "—"))

    # Executive Summary
    summary = executive_summary(df, target)
    if summary:
        st.divider()
        st.markdown("### 💼 Executive Summary")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Trend",   summary["trend"])
        s2.metric("Change",  f"{summary['change']*100:.1f}%")
        s3.metric("Risk",    summary["risk"])
        s4.metric("Average", f"{summary['avg']:,.0f}")
        s5.metric("Peak",    f"{summary['max']:,.0f}")

    # Alerts
    alerts = smart_alerts(df, target)
    if alerts:
        st.divider()
        st.markdown("### 🚨 Smart Alerts")
        for a in alerts:
            if a["type"] == "warning":
                st.warning(a["msg"])
            else:
                st.success(a["msg"])
            st.info(f"💡 **Action:** {a['action']}")

    # Key Drivers
    if st.session_state.business_insights:
        st.divider()
        st.markdown("### 📊 Key Business Drivers")
        for ins in st.session_state.business_insights:
            st.info(ins)

    # AI Narrative — auto-generates, no refresh issues
    if api_key:
        st.divider()
        st.markdown("### 🤖 AI Executive Analysis")

        # Generate if not yet generated
        if st.session_state.agent_narrative is None:
            with st.spinner("GPT-4o writing executive analysis..."):
                try:
                    st.session_state.agent_narrative = generate_agent_narrative(
                        api_key,
                        st.session_state.model_card,
                        st.session_state.business_insights or [],
                        profile,
                        st.session_state.shap_top_features or [],
                        st.session_state.problem_type,
                        st.session_state.target_col
                    )
                except Exception as e:
                    st.session_state.agent_narrative = f"AI unavailable: {e}"

        # Always show stored narrative
        st.markdown(st.session_state.agent_narrative)

        # Regenerate button — sets to None, next render auto-generates
        if st.button("🔄 Regenerate Analysis"):
            st.session_state.agent_narrative = None
            st.rerun()

# ======================================================
# CHAT — always visible after analysis
# ======================================================
st.divider()
st.markdown("### 💬 Ask DataAgentX")
st.caption("Ask anything about your data, model, or business strategy")

if not st.session_state.analyzed:
    st.info("⚠️ Run analysis first to enable intelligent chat")
else:
    # Ensure chat_history is list
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    # Quick buttons
    st.markdown("#### ⚡ Quick Insights")
    q_cols = st.columns(3)
    quick = [
        ("📉 Why did it drop?",         f"Why did {target} decrease recently?"),
        ("📊 What drives business?",     f"What are the top factors affecting {target}?"),
        ("📈 How to increase?",          f"What actions can improve {target}?"),
        ("⚠️ What are the risks?",       "What risks should I be aware of?"),
        ("🔮 Future prediction",          f"What is the likely future trend of {target}?"),
        ("💡 Strategic advice",           "Give 3 strategic recommendations based on my data.")
    ]

    question = None
    for i, (label, q) in enumerate(quick):
        if q_cols[i % 3].button(label, key=f"quick_{i}"):
            question = q

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Text input
    user_q = st.chat_input("Ask your own question...")
    if user_q:
        question = user_q

    # Process question
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = chat_with_data(
                        api_key, question,
                        st.session_state.chat_history,
                        st.session_state.model_card or {},
                        profile or {},
                        df,
                        st.session_state.problem_type,
                        st.session_state.target_col,
                        st.session_state.business_insights or []
                    )
                    st.write(resp)
                    st.session_state.chat_history.append({"role": "user",      "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
                except Exception as e:
                    st.error(f"Chat failed: {e}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ======================================================
# PDF REPORT
# ======================================================
if st.session_state.analyzed:
    st.divider()
    if st.button("📄 Download Business Report"):
        try:
            rpt_insights = st.session_state.business_insights or []
            if st.session_state.agent_narrative:
                rpt_insights = [st.session_state.agent_narrative] + rpt_insights
            path = generate_pdf_report(st.session_state.model_card or {}, rpt_insights)
            with open(path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, "DataAgentX_Report.pdf")
        except Exception as e:
            st.error(f"Report failed: {e}")
