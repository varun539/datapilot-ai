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

# ======================================================
# SESSION STATE
# ======================================================
DEFAULTS = {
    "df": None, "profile": None,
    "X": None, "y": None, "model": None,
    "problem_type": None, "target_col": None,
    "business_insights": None, "analyzed": False,
    "chat_history": [], "model_card": None,
    "agent_narrative": None, "pending_question": None,
    "processed_df": None
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🚀 DataAgentX")

uploaded = st.sidebar.file_uploader("Upload CSV")

if uploaded:
    if uploaded.size > 5 * 1024 * 1024:
        st.error("File too large (max 5MB)")
        st.stop()

    df = pd.read_csv(uploaded)
    st.session_state.df = df
    st.session_state.profile = basic_profile(df)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

if st.sidebar.button("🎯 Load Demo"):
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
    st.session_state.df = df
    st.session_state.profile = basic_profile(df)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

df = st.session_state.df
profile = st.session_state.profile

if df is None:
    st.info("👈 Upload dataset")
    st.stop()

# ======================================================
# MODE
# ======================================================
st.subheader("🎯 Choose Analysis Type")

mode = st.selectbox(
    "Analysis Mode",
    ["🧠 CEO Dashboard",
     "📊 Revenue Analysis",
     "👤 Churn Analysis",
     "🔢 Custom"]
)

# ======================================================
# CEO DASHBOARD
# ======================================================
if mode == "🧠 CEO Dashboard":

    try:
        revenue_df = adaptive_preprocess(df, "revenue")
        churn_df = adaptive_preprocess(df, "churn")

        total_revenue = revenue_df["Revenue"].sum()
        total_orders = revenue_df["Orders"].sum()
        aov = total_revenue / max(total_orders, 1)
        churn_rate = churn_df["churn"].mean()

        st.markdown("## 🧠 CEO Dashboard")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue", f"${total_revenue:,.0f}")
        c2.metric("Orders", int(total_orders))
        c3.metric("Churn Rate", f"{churn_rate*100:.1f}%")
        c4.metric("AOV", f"${aov:.2f}")

        st.divider()

        if churn_rate > 0.4:
            st.error("⚠️ High churn is hurting growth")
        elif total_revenue < revenue_df["Revenue"].median():
            st.warning("📉 Revenue unstable")
        else:
            st.success("🚀 Business healthy")

        st.markdown("### 💡 CEO Actions")
        st.success("""
👉 Reduce churn  
👉 Increase repeat purchases  
👉 Optimize pricing  
👉 Focus on high-value customers  
""")

    except Exception as e:
        st.error(f"CEO dashboard failed: {e}")

    st.stop()

# ======================================================
# PREPROCESS
# ======================================================
if mode == "📊 Revenue Analysis":
    processed_df = adaptive_preprocess(df, "revenue")
    target = "Revenue"

elif mode == "👤 Churn Analysis":
    processed_df = adaptive_preprocess(df, "churn")
    target = "churn"

else:
    processed_df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Select Target", numeric_cols)

st.session_state.target_col = target

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Analyze"):

    X, y = prepare_features(processed_df, profile, target)

    X = X.select_dtypes(include=np.number).fillna(0)
    problem = detect_problem_type(y)

    results, best_model, metrics = train_models(X, y, problem)
    model = joblib.load("models/best_model.pkl")

    # SHAP
    try:
        sample_X = X.sample(min(500, len(X)))
        explainer = shap.Explainer(model, sample_X)
        sv = explainer(sample_X)

        insights = generate_business_impact(
            sv.values, sample_X, problem, target
        )
    except:
        insights = ["Basic insights only"]

    st.session_state.update({
        "X": X,
        "y": y,
        "model": model,
        "problem_type": problem,
        "business_insights": insights,
        "analyzed": True,
        "model_card": {
            "model": best_model,
            "features": X.shape[1],
            "metrics": metrics
        },
        "processed_df": processed_df
    })

    st.success("✅ Analysis complete")

# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:

    st.subheader("🏆 Model Summary")
    st.write(st.session_state.model_card)

    st.subheader("📊 Insights")
    for i in st.session_state.business_insights:
        st.info(i)

# ======================================================
# CHAT
# ======================================================
st.subheader("💬 Ask AI")

if st.session_state.analyzed:

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    st.markdown("### ⚡ Quick Questions")

    col1, col2, col3 = st.columns(3)

    if col1.button("📉 Why drop?"):
        st.session_state.pending_question = f"Why did {target} drop?"

    if col2.button("📊 Drivers?"):
        st.session_state.pending_question = f"What drives {target}?"

    if col3.button("📈 Improve?"):
        st.session_state.pending_question = f"How to improve {target}?"

    user_input = st.chat_input("Ask...")

    if user_input:
        st.session_state.pending_question = user_input

    if st.session_state.pending_question:

        q = st.session_state.pending_question
        st.session_state.pending_question = None

        with st.chat_message("user"):
            st.write(q)

        response = chat_with_data(
            api_key,
            q,
            st.session_state.chat_history,
            st.session_state.model_card,
            profile,
            st.session_state.processed_df,
            st.session_state.problem_type,
            target,
            st.session_state.business_insights
        )

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.chat_history.append({"role": "user", "content": q})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
