import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data, generate_agent_narrative
from src.report import generate_pdf_report
from src.eda import basic_profile
from src.adaptive_preprocess import adaptive_preprocess

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataAgentX", layout="wide", page_icon="🚀")
api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

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
# SESSION STATE
# ======================================================
DEFAULTS = {
    "df": None, "profile": None,
    "X": None, "y": None, "model": None,
    "problem_type": None, "target_col": None,
    "business_insights": None, "analyzed": False,
    "chat_history": [], "model_card": None,
    "agent_narrative": None, "shap_top_features": None,
    "pending_question": None, "processed_df": None
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
        "avg": vals.mean(), "max": vals.max()
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
    return alerts

def format_metrics(metrics, problem_type):
    """Normalize metrics dict → always returns 'performance' key format"""
    if not metrics:
        return {}
    if problem_type == "regression":
        return {
            "R² (CV)":  round(metrics.get("cv_mean", metrics.get("r2", 0)), 4),
            "CV Std":   round(metrics.get("cv_std", 0), 4),
            "MAE":      round(metrics.get("mae", 0), 2),
        }
    else:
        return {
            "Accuracy (CV)": round(metrics.get("cv_mean", metrics.get("accuracy", 0)), 4),
            "CV Std":        round(metrics.get("cv_std", 0), 4),
            "F1 Score":      round(metrics.get("f1", 0), 4),
        }

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

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        if st.session_state.df is None or st.session_state.df.shape != raw.shape:
            st.session_state.df      = raw
            st.session_state.profile = basic_profile(raw)
            st.session_state.analyzed = False
            st.session_state.agent_narrative = None
            st.session_state.model_card = None
            st.session_state.chat_history = []
            st.session_state.processed_df = None
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

if st.sidebar.button("🎯 Walmart Demo"):
    try:
        demo = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
        st.session_state.df           = demo
        st.session_state.profile      = basic_profile(demo)
        st.session_state.analyzed     = False
        st.session_state.agent_narrative = None
        st.session_state.model_card   = None
        st.session_state.chat_history = []
        st.session_state.processed_df = None
    except Exception as e:
        st.sidebar.error(f"Demo failed: {e}")

df      = st.session_state.df
profile = st.session_state.profile

if df is not None:
    st.sidebar.success("✅ Dataset Ready")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Cols", df.shape[1])

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
# ANALYSIS MODE
# ======================================================
st.markdown("### 🎯 What do you want to analyze?")

mode = st.selectbox(
    "Choose Analysis Type",
    ["📊 Revenue / Sales Forecasting",
     "👤 Customer Churn Prediction",
     "🔢 Custom — I'll pick the target"]
)

# ======================================================
# PROCESS DATA BASED ON MODE
# ======================================================
if mode == "📊 Revenue / Sales Forecasting":
    try:
        processed_df = adaptive_preprocess(df, mode="revenue")
        target = "Revenue"
        st.success(f"✅ Revenue data ready — {processed_df.shape[0]} rows")
    except Exception as e:
        st.warning(f"Auto-preprocess failed ({e}) — using raw data")
        processed_df = df.copy()
        # Find best revenue-like column
        priority = ["Weekly_Sales","Sales","Revenue","Profit","Price"]
        target = next((c for c in priority if c in df.columns),
                      df.select_dtypes(include="number").columns[0])

elif mode == "👤 Customer Churn Prediction":
    try:
        processed_df = adaptive_preprocess(df, mode="churn")
        target = "Churn"
        st.success(f"✅ Churn data ready — {processed_df.shape[0]} customers")
    except Exception as e:
        st.warning(f"Auto-preprocess failed ({e}) — using raw data")
        processed_df = df.copy()
        target = "Churn" if "Churn" in df.columns else df.columns[-1]

else:
    processed_df = df.copy()
    # Manual target selection
    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.error("No numeric columns found!")
        st.stop()
    priority = ["Weekly_Sales","Sales","Revenue","Profit"]
    default = next((c for c in priority if c in numeric_cols), numeric_cols[0])
    target = st.selectbox("Select Target Column", numeric_cols,
                          index=numeric_cols.index(default))

st.session_state.target_col = target

# Verify target exists
if target not in processed_df.columns:
    st.error(f"Target '{target}' not found after processing!")
    st.stop()

# ======================================================
# ANALYZE BUTTON
# ======================================================
if st.button("🚀 Analyze", use_container_width=False):
    with st.spinner("Training models... 2-3 minutes"):
        try:
            proc_profile = basic_profile(processed_df)

            # Pipeline returns X, y
            X, y = prepare_features(
                processed_df, proc_profile, target, training=True
            )

            # Safety
            X = X.select_dtypes(include=[np.number]).fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            y = y.reset_index(drop=True)
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len].reset_index(drop=True)
            y = y.iloc[:min_len].reset_index(drop=True)

            if X.shape[1] == 0:
                st.error("No features after processing!")
                st.stop()

            # Train — returns (df_results, best_name, metrics_dict)
            problem = detect_problem_type(y)
            results, best_model_name, raw_metrics = train_models(X, y, problem)
            model = joblib.load("models/best_model.pkl")

            # Normalize metrics → always "performance" format
            performance = format_metrics(raw_metrics, problem)

            # SHAP
            shap_top = []
            insights = []
            try:
                sample_X = X.sample(min(500, len(X)), random_state=42)
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(sample_X)
                if isinstance(sv, list):
                    sv = sv[1]
                mean_abs = np.abs(sv).mean(axis=0)
                top_idx  = np.argsort(mean_abs)[::-1][:5]
                shap_top = [(sample_X.columns[i], round(float(mean_abs[i]),4)) for i in top_idx]
                insights = generate_business_impact(sv, sample_X, problem, target)
            except Exception as se:
                insights = [f"Analysis complete. Detailed SHAP skipped: {se}"]

            # Build model_card — unified format
            model_card = {
                "model":       best_model_name,
                "problem":     problem,
                "rows":        processed_df.shape[0],
                "features":    X.shape[1],
                "target":      target,
                "performance": performance,  # ← always "performance" key!
                "metrics":     raw_metrics,  # ← raw for technical view
            }

            st.session_state.update({
                "X": X, "y": y, "model": model,
                "problem_type":      problem,
                "target_col":        target,
                "business_insights": insights,
                "analyzed":          True,
                "model_card":        model_card,
                "shap_top_features": shap_top,
                "agent_narrative":   None,
                "processed_df":      processed_df,
                "feature_schema":    X.columns.tolist(),
            })

            st.success(f"✅ {best_model_name} trained!")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)

# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:
    mc   = st.session_state.model_card or {}
    perf = mc.get("performance", {})
    disp_df = st.session_state.processed_df or df

    # Model Performance
    st.divider()
    st.markdown("### 🏆 Model Performance")
    cols_m = st.columns(len(perf) + 2)
    cols_m[0].metric("Best Model", mc.get("model", "—"))
    cols_m[1].metric("Features",   mc.get("features", "—"))
    for i, (k, v) in enumerate(perf.items()):
        cols_m[i+2].metric(k, f"{v}")

    # Executive Summary
    summary = executive_summary(disp_df, target)
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
    alerts = smart_alerts(disp_df, target)
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

    # AI Narrative — auto generates, no refresh
    if api_key:
        st.divider()
        st.markdown("### 🤖 AI Executive Analysis")
        if st.session_state.agent_narrative is None:
            with st.spinner("GPT-4o writing analysis..."):
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

        st.markdown(st.session_state.agent_narrative)

        if st.button("🔄 Regenerate Analysis"):
            st.session_state.agent_narrative = None
            st.rerun()

    # Technical Details
    with st.expander("⚙️ Technical Details"):
        raw_m = mc.get("metrics", {})
        st.markdown("### 📈 Full Metrics")
        if st.session_state.problem_type == "regression":
            st.write(f"R²:      {round(raw_m.get('r2', 0), 4)}")
            st.write(f"MAE:     {round(raw_m.get('mae', 0), 2)}")
            st.write(f"RMSE:    {round(raw_m.get('rmse', 0), 2)}")
            st.write(f"CV Mean: {round(raw_m.get('cv_mean', 0), 4)}")
            st.write(f"CV Std:  {round(raw_m.get('cv_std', 0), 4)}")
        else:
            st.write(f"Accuracy: {round(raw_m.get('accuracy', 0), 4)}")
            st.write(f"F1:       {round(raw_m.get('f1', 0), 4)}")
            st.write(f"CV Mean:  {round(raw_m.get('cv_mean', 0), 4)}")
            st.write(f"CV Std:   {round(raw_m.get('cv_std', 0), 4)}")

# ======================================================
# CHAT
# ======================================================
st.divider()
st.markdown("### 💬 Ask DataAgentX")
st.caption("Ask anything about your data, model, or business strategy")

if not st.session_state.analyzed:
    st.info("⚠️ Run analysis first to enable chat")
else:
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    disp_df = st.session_state.processed_df or df

    # Quick buttons using pending_question pattern (no refresh bug!)
    st.markdown("#### ⚡ Quick Insights")
    q_cols = st.columns(3)
    quick = [
        ("📉 Why did it drop?",       f"Why did {target} decrease recently?"),
        ("📊 What drives business?",  f"What are the top factors affecting {target}?"),
        ("📈 How to increase?",       f"What actions can improve {target}?"),
        ("⚠️ What are the risks?",    "What risks should I watch out for?"),
        ("🔮 Future prediction",       f"What is the likely future trend of {target}?"),
        ("💡 Strategic advice",        "Give 3 concrete actions to improve my business.")
    ]
    for i, (label, q) in enumerate(quick):
        if q_cols[i % 3].button(label, key=f"q_{i}"):
            st.session_state.pending_question = q

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask your own question...")
    if user_input:
        st.session_state.pending_question = user_input

    # Process pending question
    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None

        with st.chat_message("user"):
            st.write(q)

        st.session_state.chat_history.append({"role": "user", "content": q})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = chat_with_data(
                        api_key, q,
                        st.session_state.chat_history.copy(),
                        st.session_state.model_card or {},
                        profile or {},
                        disp_df,
                        st.session_state.problem_type,
                        st.session_state.target_col,
                        st.session_state.business_insights or []
                    )
                    st.write(resp)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
                except Exception as e:
                    st.error(f"Chat error: {e}")

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
            rpt = st.session_state.business_insights or []
            if st.session_state.agent_narrative:
                rpt = [st.session_state.agent_narrative] + rpt
            path = generate_pdf_report(
                st.session_state.model_card or {}, rpt
            )
            with open(path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, "DataAgentX_Report.pdf")
        except Exception as e:
            st.error(f"Report failed: {e}")
