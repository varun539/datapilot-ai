import streamlit as st
import pandas as pd
import numpy as np
import shap
import os
import joblib

from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data
from src.eda import basic_profile
from src.adaptive_preprocess import adaptive_preprocess
from src.leakage import detect_leakage

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataPilot AI", layout="wide")
api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

# ======================================================
# SESSION STATE
# ======================================================
DEFAULTS = {
    "df": None,
    "profile": None,
    "model": None,
    "problem_type": None,
    "target_col": None,
    "business_insights": [],
    "model_card": {},
    "processed_df": None,
    "analyzed": False,
    "chat_history": []
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# HEADER
# ======================================================
st.title("🚀 DataPilot AI — Agentic AutoML Platform")
st.caption("Upload your data → Train models → Get AI-powered insights")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("📂 Upload Data")

file = st.sidebar.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    if st.session_state.df is None or st.session_state.df.shape != df.shape:
        st.session_state.df = df
        st.session_state.profile = basic_profile(df)
        st.session_state.analyzed = False
        st.session_state.chat_history = []

if st.sidebar.button("📊 Load Demo Dataset"):
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
    st.session_state.df = df
    st.session_state.profile = basic_profile(df)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

df = st.session_state.df
profile = st.session_state.profile

if df is None:
    st.warning("Upload a dataset to begin.")
    st.stop()

# ======================================================
# DATA PREVIEW
# ======================================================
st.subheader("📄 Data Preview")
st.dataframe(df.head())

# ======================================================
# TARGET SELECTION
# ======================================================
st.subheader("🎯 Target Selection")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found.")
    st.stop()

priority = ["Revenue", "Weekly_Sales", "Sales", "Profit"]
default = next((c for c in priority if c in numeric_cols), numeric_cols[0])

target = st.selectbox(
    "Select Target Column",
    numeric_cols,
    index=numeric_cols.index(default)
)

st.session_state.target_col = target

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Run Analysis"):

    with st.spinner("Running full ML pipeline..."):

        try:
            # -------------------------
            # PREPROCESS
            # -------------------------
            try:
                processed_df = adaptive_preprocess(df, "revenue")
                if target not in processed_df.columns:
                    processed_df = df.copy()
            except:
                processed_df = df.copy()

            proc_profile = basic_profile(processed_df)

            # -------------------------
            # PIPELINE
            # -------------------------
            X, y = prepare_features(processed_df, proc_profile, target)

            if X.empty or y.empty:
                st.error("❌ Feature engineering failed. Check dataset.")
                st.stop()

            X = X.select_dtypes(include=np.number).fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            # -------------------------
            # 🚨 LEAKAGE CHECK (FIXED POSITION)
            # -------------------------
            warnings, high_risk = detect_leakage(X, y)

            st.subheader("🛡️ Data Leakage Check")

            for w in warnings:
                if "HIGH RISK" in w:
                    st.error(w)
                elif "Suspicious" in w:
                    st.warning(w)
                else:
                    st.success(w)

            if high_risk:
                st.error("❌ Training stopped due to data leakage")
                st.stop()

            # -------------------------
            # MODEL TRAINING
            # -------------------------
            problem = detect_problem_type(y)

            results_df, model_name, metrics = train_models(X, y, problem)

            model = joblib.load("models/best_model.pkl")

            # -------------------------
            # SHAP
            # -------------------------
            insights = []
            try:
                sample_X = X.sample(min(300, len(X)), random_state=42)

                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(sample_X)

                if isinstance(sv, list):
                    sv = sv[1]

                insights = generate_business_impact(
                    sv, sample_X, problem, target
                )

            except Exception as e:
                insights = [f"SHAP skipped: {e}"]

            # -------------------------
            # SAVE STATE
            # -------------------------
            st.session_state.update({
                "model": model,
                "problem_type": problem,
                "business_insights": insights,
                "processed_df": processed_df,
                "analyzed": True,
                "chat_history": [],
                "model_card": {
                    "model": model_name,
                    "features": X.shape[1],
                    "rows": processed_df.shape[0],
                    "target": target,
                    "metrics": metrics
                }
            })

            st.success(f"✅ {model_name} trained successfully!")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:

    mc = st.session_state.model_card
    metrics = mc.get("metrics", {})
    hold = metrics.get("holdout", {})
    cv = metrics.get("cv", {})

    st.divider()
    st.subheader("🏆 Model Summary")

    c1, c2 = st.columns(2)
    c1.metric("Model", mc.get("model"))
    c2.metric("Features", mc.get("features"))

    st.subheader("📈 Metrics")

    if st.session_state.problem_type == "regression":
        st.write(f"R²: {round(hold.get('r2', 0), 4)}")
        st.write(f"MAE: {round(hold.get('mae', 0), 2)}")
        st.write(f"RMSE: {round(hold.get('rmse', 0), 2)}")
    else:
        st.write(f"Accuracy: {round(hold.get('accuracy', 0), 4)}")
        st.write(f"F1: {round(hold.get('f1', 0), 4)}")

    st.write(f"CV Mean: {round(cv.get('mean', 0), 4)}")
    st.write(f"CV Std: {round(cv.get('std', 0), 4)}")

    # -------------------------
    # INSIGHTS
    # -------------------------
    st.subheader("📊 Business Insights")

    for ins in st.session_state.business_insights:
        st.info(ins)

# ======================================================
# CHAT
# ======================================================
st.divider()
st.subheader("💬 Ask Questions About Your Data")

if not st.session_state.analyzed:
    st.info("Run analysis first to enable chat")
else:

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about your business...")

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
                    profile or {},
                    st.session_state.processed_df,
                    st.session_state.problem_type,
                    st.session_state.target_col,
                    st.session_state.business_insights
                )
                st.write(response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

# ======================================================
# TECHNICAL
# ======================================================
with st.expander("⚙️ Technical Details"):

    if st.session_state.analyzed:

        st.markdown("### Pipeline")
        st.code("""
Upload → Preprocess → Feature Engineering → AutoML → SHAP → Insights
""")

        st.json(st.session_state.model_card)

    else:
        st.info("Run analysis to view details")
