import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data
from src.eda import basic_profile
from src.adaptive_preprocess import adaptive_preprocess

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataAgentX", layout="wide")
api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

# ======================================================
# SESSION STATE
# ======================================================
DEFAULTS = {
    "df": None, "profile": None,
    "X": None, "y": None, "model": None,
    "problem_type": None, "target_col": None,
    "business_insights": None, "analyzed": False,
    "chat_history": [], "model_card": {},
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
    st.info("👈 Upload dataset to start")
    st.stop()

# ======================================================
# MODE
# ======================================================
st.subheader("🎯 Choose Analysis Type")

mode = st.selectbox(
    "Analysis Mode",
    ["🧠 CEO Dashboard", "📊 Revenue Analysis", "👤 Churn Analysis", "🔢 Custom"]
)

# ======================================================
# CEO DASHBOARD
# ======================================================
if mode == "🧠 CEO Dashboard":

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
        insights = ["Basic insights available"]

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
# 💬 CHAT (NO REFRESH BUG)
# ======================================================
st.subheader("💬 Ask AI")

if st.session_state.analyzed:

    # show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about your business")

    if user_input:

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_data(
                    api_key,
                    user_input,
                    st.session_state.chat_history[:-1],
                    st.session_state.model_card,
                    profile,
                    st.session_state.processed_df,
                    st.session_state.problem_type,
                    target,
                    st.session_state.business_insights
                )
                st.write(response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

# ======================================================
# ⚙️ TECHNICAL VIEW (HIDDEN)
# ======================================================
with st.expander("⚙️ Technical Details"):

    st.markdown("### 🧠 Pipeline")
    st.code("""
Upload → Preprocess → Feature Engineering → AutoML → CV → SHAP → Insights → Chat
""")

    st.markdown("### 🤖 Model Info")
    st.write(st.session_state.model_card)

    if st.session_state.model_card:

        metrics = st.session_state.model_card.get("metrics", {})

        if st.session_state.problem_type == "regression":
            st.write("R²:", round(metrics.get("r2", 0), 4))
            st.write("RMSE:", round(metrics.get("rmse", 0), 2))
            st.write("MAE:", round(metrics.get("mae", 0), 2))
        else:
            st.write("Accuracy:", round(metrics.get("accuracy", 0), 4))
            st.write("F1:", round(metrics.get("f1", 0), 4))

        st.write("CV Mean:", round(metrics.get("cv_mean", 0), 4))
        st.write("CV Std:", round(metrics.get("cv_std", 0), 4))

    else:
        st.info("Run analysis to see metrics")
