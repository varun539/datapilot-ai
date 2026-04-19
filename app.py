import streamlit as st
import pandas as pd
import numpy as np
import shap
import os

# from src.pipeline import prepare_features
# from src.automl import detect_problem_type, train_models
# from src.impact import generate_business_impact
# from src.agent import chat_with_data
# from src.eda import basic_profile
# from src.adaptive_preprocess import adaptive_preprocess


from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data
from src.eda import basic_profile
from src.adaptive_preprocess import adaptive_preprocess

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
    "analyzed": False
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# HEADER
# ======================================================
st.title("🚀 DataPilot AI — Agentic AutoML Platform")
st.caption("Upload your data → Train models → Get AI-powered insights")

st.info("✅ This pipeline avoids data leakage using proper feature isolation and evaluation strategy.")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("📂 Upload Data")

file = st.sidebar.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    st.session_state.df = df
    st.session_state.profile = basic_profile(df)
    st.session_state.analyzed = False

if st.sidebar.button("📊 Load Demo Dataset"):
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
    st.session_state.df = df
    st.session_state.profile = basic_profile(df)
    st.session_state.analyzed = False

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

target = st.selectbox("Select Target Column", numeric_cols)

st.session_state.target_col = target

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Run Analysis"):

    with st.spinner("Training models..."):

        try:
            processed_df = adaptive_preprocess(df, "revenue")

            X, y = prepare_features(processed_df, profile, target)
            X = X.select_dtypes(include=np.number).fillna(0)

            problem = detect_problem_type(y)

            model, model_name, metrics = train_models(X, y, problem)

            # SHAP
            try:
                sample_X = X.sample(min(300, len(X)))
                explainer = shap.Explainer(model, sample_X)
                sv = explainer(sample_X)

                insights = generate_business_impact(
                    sv.values, sample_X, problem, target
                )
            except:
                insights = ["SHAP analysis unavailable"]

            st.session_state.update({
                "model": model,
                "problem_type": problem,
                "business_insights": insights,
                "processed_df": processed_df,
                "model_card": {
                    "model": model_name,
                    "features": X.shape[1],
                    "metrics": metrics
                },
                "analyzed": True
            })

            st.success("Analysis complete")

        except Exception as e:
            st.error(f"Error: {e}")

# ======================================================
# RESULTS
# ======================================================
# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:

    mc = st.session_state.model_card
    metrics = mc.get("metrics", {})

    hold = metrics.get("holdout", {})
    cv   = metrics.get("cv", {})

    st.divider()
    st.subheader("🏆 Model Summary")

    c1, c2 = st.columns(2)
    c1.metric("Model", mc.get("model", "-"))
    c2.metric("Features", mc.get("features", "-"))

    st.subheader("📈 Metrics")

    if st.session_state.problem_type == "regression":
        st.write(f"R²: {round(hold.get('r2', 0), 4)}")
        st.write(f"MAE: {round(hold.get('mae', 0), 2)}")
        st.write(f"RMSE: {round(hold.get('rmse', 0), 2)}")
    else:
        st.write(f"Accuracy: {round(hold.get('accuracy', 0), 4)}")
        st.write(f"F1 Score: {round(hold.get('f1', 0), 4)}")

    st.write(f"CV Mean: {round(cv.get('mean', 0), 4)}")
    st.write(f"CV Std: {round(cv.get('std', 0), 4)}")

    # INSIGHTS
    st.subheader("📊 Business Insights")
    for ins in st.session_state.business_insights:
        st.info(ins)

    # ======================================================
    # INSIGHTS
    # ======================================================
    st.subheader("📊 Business Insights")

    for ins in st.session_state.business_insights:
        st.info(ins)

# ======================================================
# SIMPLE CHAT (RECRUITER SAFE)
# ======================================================
st.divider()
st.subheader("💬 Ask Questions About Your Data")

if st.session_state.analyzed:

    user_input = st.text_input("Ask a question")

    if user_input:

        response = chat_with_data(
            api_key,
            user_input,
            [],
            st.session_state.model_card,
            profile,
            st.session_state.processed_df,
            st.session_state.problem_type,
            st.session_state.target_col,
            st.session_state.business_insights
        )

        st.write(response)

# ======================================================
# TECHNICAL VIEW
# ======================================================
with st.expander("⚙️ Technical Details"):

    st.code("""
Pipeline:
1. Data cleaning
2. Feature engineering
3. Train/test split
4. AutoML training
5. Cross-validation
6. SHAP explainability
""")

    st.json(st.session_state.model_card)
