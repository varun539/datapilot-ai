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
from src.agent import chat_with_data, suggest_target_column, generate_agent_narrative, diagnose_dataset
from src.report import generate_pdf_report
from src.eda import basic_profile

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="DataAgentX — AI Business Intelligence",
    layout="wide",
    page_icon="🚀"
)

api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

# ======================================================
# CUSTOM CSS — Premium Dark Theme
# ======================================================
st.markdown("""
<style>
/* Dark premium feel */
[data-testid="stAppViewContainer"] {
    background: #0a0d14;
}
[data-testid="stSidebar"] {
    background: #0f1320;
    border-right: 1px solid #1e2535;
}
/* Metric cards */
[data-testid="stMetric"] {
    background: #141826;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 16px !important;
}
/* Info boxes */
.stAlert {
    border-radius: 10px;
}
/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}
/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE
# ======================================================
STATE_KEYS = [
    "df", "profile", "X", "y", "model",
    "problem_type", "target_col",
    "business_insights", "analyzed",
    "chat_history", "model_card",
    "agent_narrative", "shap_top_features",
    "cv_score", "residual_std", "feature_schema"
]
for key in STATE_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.analyzed is None:
    st.session_state.analyzed = False
if st.session_state.chat_history is None:
    st.session_state.chat_history = []
if "trigger_narrative" not in st.session_state:
    st.session_state.trigger_narrative = False

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def generate_executive_summary(df, target):
    if df is None or target not in df.columns:
        return {}
    vals = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(vals) < 10:
        return {}
    recent = vals.tail(int(len(vals) * 0.2)).mean()
    prev   = vals.head(int(len(vals) * 0.8)).mean()
    change = (recent - prev) / (abs(prev) + 1e-6)
    slope  = np.polyfit(range(len(vals)), vals, 1)[0]
    trend  = "📈 Growing"   if slope > 0 else "📉 Declining"
    risk   = "🔴 High"      if change < -0.1 else "🟡 Medium" if change < -0.05 else "🟢 Low"
    return {"trend": trend, "change": change, "risk": risk,
            "avg": vals.mean(), "max": vals.max(), "min": vals.min()}

def generate_alerts(df, target):
    if df is None or target not in df.columns:
        return []
    vals = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(vals) < 20:
        return []
    recent = vals.tail(int(len(vals) * 0.2)).mean()
    prev   = vals.head(int(len(vals) * 0.8)).mean()
    change = (recent - prev) / (abs(prev) + 1e-6)
    alerts = []
    if change < -0.1:
        alerts.append({
            "type": "warning",
            "msg":  f"⚠️ {target} dropped {abs(change)*100:.1f}% vs historical average",
            "action": "Consider increasing promotions, reviewing pricing strategy, or investigating supply chain issues."
        })
    elif change > 0.1:
        alerts.append({
            "type": "success",
            "msg":  f"🚀 {target} grew {abs(change)*100:.1f}% vs historical average",
            "action": "Capitalize by scaling inventory, increasing ad spend, and expanding to new markets."
        })

    # Outlier detection
    z_scores = np.abs((vals - vals.mean()) / (vals.std() + 1e-6))
    outlier_pct = (z_scores > 3).mean()
    if outlier_pct > 0.02:
        alerts.append({
            "type": "warning",
            "msg":  f"⚠️ {outlier_pct*100:.1f}% of {target} values are statistical outliers",
            "action": "Investigate data quality or identify exceptional events driving extreme values."
        })
    return alerts

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown("## 🚀 DataAgentX")
st.sidebar.caption("AI Business Intelligence Platform")

# API Status
with st.sidebar.expander("🔑 AI Status", expanded=False):
    if api_key:
        st.success("✅ AI Connected")
    else:
        st.warning("⚠️ No API key — AI features disabled")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Data Source")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if st.sidebar.button("🎯 Use Walmart Demo"):
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
        profile = basic_profile(df)
        st.session_state.update({
            "df": df, "profile": profile,
            "analyzed": False, "target_col": None,
            "agent_narrative": None, "chat_history": []
        })
        st.success("✅ Demo loaded!")
    except:
        st.error("Demo load failed — upload CSV instead")

if file:
    try:
        df = pd.read_csv(file)
        profile = basic_profile(df)
        st.session_state.update({
            "df": df, "profile": profile,
            "analyzed": False, "target_col": None,
            "agent_narrative": None, "chat_history": []
        })
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

df      = st.session_state.get("df")
profile = st.session_state.get("profile")

# Dataset info in sidebar
if df is not None:
    st.sidebar.success("✅ Dataset Ready")
    st.sidebar.metric("Rows",    df.shape[0])
    st.sidebar.metric("Columns", df.shape[1])

    missing = df.isnull().sum().sum()
    if missing > 0:
        st.sidebar.warning(f"⚠️ {missing} missing values")
    else:
        st.sidebar.success("✅ No missing values")

# ======================================================
# MAIN HEADER
# ======================================================
st.markdown("# 🚀 DataAgentX")
st.markdown("**Upload data → AI analyzes → Get business decisions**")
st.divider()

if df is None:
    st.info("👈 Upload a CSV or use the Walmart demo from the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Models Available", "5")
    col2.metric("AI Engine", "GPT-4o")
    col3.metric("Explainability", "SHAP")
    st.stop()

# ======================================================
# TARGET SELECTION — SMART
# ======================================================
st.markdown("### 🎯 What do you want to predict?")

# Smart target detection
if st.session_state.target_col is None:
    priority = ["Weekly_Sales","Sales","Revenue","Profit","Price","Target","Churn"]
    auto_target = next((c for c in priority if c in df.columns), df.columns[-1])
    st.session_state.target_col = auto_target

# Show ALL numeric columns — let user choose freely
# Only skip pure ID columns (Store, Row_ID etc)
id_only_kw = ["row_id", "customerid", "orderid"]
numeric_targets = []
for c in df.columns:
    cl = c.lower().replace(" ", "_")
    if cl in id_only_kw:
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_targets.append(c)

if not numeric_targets:
    numeric_targets = df.select_dtypes(include="number").columns.tolist()

# Smart default priority
priority = ["Weekly_Sales", "Sales", "Revenue", "Profit", "Price", "Target"]
default_col = next(
    (c for c in priority if c in numeric_targets),
    st.session_state.target_col if st.session_state.target_col in numeric_targets
    else numeric_targets[0]
)
default_idx = numeric_targets.index(default_col)

target = st.selectbox("Select Target Column", numeric_targets, index=default_idx)
st.session_state.target_col = target

# ======================================================
# ANALYZE BUTTON
# ======================================================
col_btn1, col_btn2 = st.columns([1, 4])
run_analysis = col_btn1.button("🚀 Analyze", use_container_width=True)

if run_analysis:
    with st.spinner("Training models and generating insights..."):
        try:
            # Pipeline now returns X, y together
            X, y = prepare_features(df, profile, target, training=True)

            # Safety checks
            if X is None or X.shape[1] == 0:
                st.error("No features available after processing. Check your dataset.")
                st.stop()

            if y is None or len(y) == 0:
                st.error("Target column has no valid values.")
                st.stop()

            # Drop any remaining non-numeric columns
            X = X.select_dtypes(include=[np.number]).fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            problem = detect_problem_type(y)
            results, best_model_name = train_models(X, y, problem)
            model = joblib.load("models/best_model.pkl")

            # Cross validation
            if problem == "regression":
                cv    = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                cv_label = "R²"
            else:
                cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                cv_label = "Accuracy"

            cv_mean = scores.mean()
            cv_std  = scores.std()

            residual_std = None
            if problem == "regression":
                preds = model.predict(X)
                residual_std = float(np.std(y - preds))

            # SHAP
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                mean_abs  = np.abs(shap_vals).mean(axis=0)
                top_idx   = np.argsort(mean_abs)[::-1][:5]
                shap_top  = [(X.columns[i], round(float(mean_abs[i]), 4)) for i in top_idx]
                insights  = generate_business_impact(shap_vals, X, problem, target)
            except Exception as e:
                shap_top = []
                insights = [f"Model trained successfully. SHAP skipped: {e}"]

            model_card = {
                "model": best_model_name,
                "problem": problem,
                "rows": df.shape[0],
                "features": X.shape[1],
                "target": target,
                "performance": {
                    f"{cv_label} (CV)": round(cv_mean, 4),
                    "CV Std": round(cv_std, 4)
                }
            }

            st.session_state.update({
                "X": X, "y": y, "model": model,
                "problem_type": problem,
                "target_col": target,
                "business_insights": insights,
                "analyzed": True,
                "model_card": model_card,
                "shap_top_features": shap_top,
                "cv_score": cv_mean,
                "residual_std": residual_std,
                "feature_schema": X.columns.tolist(),
                "agent_narrative": None
            })

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

# ======================================================
# RESULTS — Only show after analysis
# ======================================================
if st.session_state.analyzed:

    mc = st.session_state.model_card or {}

    st.divider()
    st.markdown("### 🏆 Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",  mc.get("model", "—"))
    perf = mc.get("performance", {})
    for i, (k, v) in enumerate(perf.items()):
        [c2, c3][i].metric(k, f"{v:.4f}")
    c4.metric("Features Used", mc.get("features", "—"))

    # ── Executive Summary ──
    summary = generate_executive_summary(df, target)
    if summary:
        st.divider()
        st.markdown("### 💼 Executive Summary")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Trend",   summary["trend"])
        s2.metric("Change",  f"{summary['change']*100:.1f}%")
        s3.metric("Risk",    summary["risk"])
        s4.metric("Average", f"{summary['avg']:,.0f}")
        s5.metric("Peak",    f"{summary['max']:,.0f}")

    # ── Alerts ──
    alerts = generate_alerts(df, target)
    if alerts:
        st.divider()
        st.markdown("### 🚨 Smart Alerts")
        for a in alerts:
            if a["type"] == "warning":
                st.warning(a["msg"])
            else:
                st.success(a["msg"])
            st.info(f"💡 **Recommended Action:** {a['action']}")

    # ── SHAP Key Drivers ──
    if st.session_state.business_insights:
        st.divider()
        st.markdown("### 📊 Key Business Drivers")
        for insight in st.session_state.business_insights:
            st.info(insight)

    # ── AI Narrative ──
    if api_key:
        st.divider()
        st.markdown("### 🤖 AI Executive Analysis")

        # Auto-generate only when:
        # 1. First time after analysis (narrative is None)
        # 2. User clicked Refresh button (trigger_narrative = True)
        should_generate = (
            st.session_state.agent_narrative is None or
            st.session_state.trigger_narrative
        )

        if should_generate:
            st.session_state.trigger_narrative = False
            with st.spinner("GPT-4o generating executive analysis..."):
                try:
                    narrative = generate_agent_narrative(
                        api_key,
                        st.session_state.model_card,
                        st.session_state.business_insights or [],
                        profile,
                        st.session_state.shap_top_features or [],
                        st.session_state.problem_type,
                        st.session_state.target_col
                    )
                    st.session_state.agent_narrative = narrative
                except Exception as e:
                    st.session_state.agent_narrative = f"AI analysis unavailable: {e}"

        # Always display stored narrative — no rerun needed
        if st.session_state.agent_narrative:
            st.markdown(st.session_state.agent_narrative)

            col_r1, col_r2 = st.columns([1, 5])
            if col_r1.button("🔄 Refresh Analysis", key="refresh_narrative"):
                st.session_state.agent_narrative = None
                st.session_state.trigger_narrative = True

# ======================================================
# SMART CHAT
# ======================================================
st.divider()
st.markdown("### 💬 Ask DataAgentX")
st.caption("Ask anything about your data, model, or business strategy")

if not st.session_state.analyzed:
    st.info("⚠️ Run analysis first to enable intelligent chat")
else:
    # Quick insight buttons
    st.markdown("#### ⚡ Quick Insights")
    cols = st.columns(3)
    quick_questions = [
        ("📉 Why did it drop?",        f"Why did {target} decrease recently? Give data-backed reasons."),
        ("📊 What drives my business?", f"What are the top factors affecting {target} and why?"),
        ("📈 How to increase?",         f"Based on the data, what actions can improve {target}?"),
        ("⚠️ What are the risks?",      "What risks or negative trends should I be aware of?"),
        ("🔮 Future prediction",         f"What is the likely future trend of {target}?"),
        ("💡 Strategic advice",          "Give 3 strategic recommendations based on my data.")
    ]

    question = None
    for i, (label, q) in enumerate(quick_questions):
        if cols[i % 3].button(label, key=f"q_{i}"):
            question = q

    # Custom input
    user_input = st.chat_input("Ask your own question about the data...")
    if user_input:
        question = user_input

    # Display chat history
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Process question
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = chat_with_data(
                        api_key, question,
                        st.session_state.chat_history,
                        st.session_state.model_card or {},
                        profile or {},
                        df,
                        st.session_state.problem_type,
                        st.session_state.target_col,
                        st.session_state.business_insights or []
                    )
                    st.write(response)
                    st.session_state.chat_history.append({"role": "user",      "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
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
            insights_for_report = st.session_state.business_insights or []
            if st.session_state.agent_narrative:
                insights_for_report = [st.session_state.agent_narrative] + insights_for_report

            path = generate_pdf_report(
                st.session_state.model_card or {},
                insights_for_report
            )
            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report",
                    f,
                    "DataAgentX_Report.pdf"
                )
        except Exception as e:
            st.error(f"Report failed: {e}")
