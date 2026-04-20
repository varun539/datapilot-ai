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

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataPilot AI", layout="wide")
api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

# ======================================================
# SESSION STATE
# ======================================================
DEFAULTS = {
    "df": None, "profile": None,
    "model": None, "problem_type": None,
    "target_col": None, "business_insights": [],
    "model_card": {}, "processed_df": None,
    "analyzed": False, "chat_history": []
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
        st.session_state.df       = df
        st.session_state.profile  = basic_profile(df)
        st.session_state.analyzed = False
        st.session_state.chat_history = []

if st.sidebar.button("📊 Load Demo Dataset"):
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
    st.session_state.df       = df
    st.session_state.profile  = basic_profile(df)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

df      = st.session_state.df
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
default  = next((c for c in priority if c in numeric_cols), numeric_cols[0])
target   = st.selectbox("Select Target Column", numeric_cols,
                        index=numeric_cols.index(default))
st.session_state.target_col = target

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Run Analysis"):
    with st.spinner("Training models..."):
        try:
            # Preprocess
            try:
                processed_df = adaptive_preprocess(df, "revenue")
                # Make sure target exists after preprocess
                if target not in processed_df.columns:
                    processed_df = df.copy()
            except Exception:
                processed_df = df.copy()

            proc_profile = basic_profile(processed_df)

            # Pipeline — returns (X, y)
            X, y = prepare_features(processed_df, proc_profile, target, training=True)
            X = X.select_dtypes(include=np.number).fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            # Align lengths
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len].reset_index(drop=True)
            y = y.iloc[:min_len].reset_index(drop=True)

            if X.shape[1] == 0:
                st.error("No features available. Check your dataset.")
                st.stop()

            problem = detect_problem_type(y)

            # train_models returns (results_df, best_name, metrics_dict)
            results_df, model_name, metrics = train_models(X, y, problem)

            # Safe model load
            model_path = "models/best_model.pkl"
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                st.error("Model save failed. Please try again.")
                st.stop()
            model = joblib.load(model_path)

            # SHAP
            insights = []
            try:
                sample_X  = X.sample(min(300, len(X)), random_state=42)
                explainer = shap.TreeExplainer(model)
                sv        = explainer.shap_values(sample_X)
                if isinstance(sv, list):
                    sv = sv[1]
                insights = generate_business_impact(sv, sample_X, problem, target)
            except Exception as se:
                insights = [f"Analysis complete. SHAP skipped: {se}"]

            # Normalize metrics
            cv_m = metrics.get("cv", {})
            ho_m = metrics.get("holdout", {})

            st.session_state.update({
                "model":             model,
                "problem_type":      problem,
                "business_insights": insights,
                "processed_df":      processed_df,
                "chat_history":      [],
                "analyzed":          True,
                "model_card": {
                    "model":    model_name,
                    "features": X.shape[1],
                    "rows":     processed_df.shape[0],
                    "target":   target,
                    "metrics":  metrics,
                    # unified display format
                    "performance": {
                        "R² (CV)" if problem=="regression" else "Accuracy (CV)":
                            round(cv_m.get("mean", 0), 4),
                        "CV Std": round(cv_m.get("std", 0), 4),
                        "MAE" if problem=="regression" else "F1":
                            round(ho_m.get("mae", ho_m.get("f1", 0)), 2),
                    }
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

    mc   = st.session_state.model_card
    perf = mc.get("performance", {})

    st.divider()
    st.subheader("🏆 Model Summary")

    cols_m = st.columns(len(perf) + 2)
    cols_m[0].metric("Model",    mc.get("model", "—"))
    cols_m[1].metric("Features", mc.get("features", "—"))
    for i, (k, v) in enumerate(perf.items()):
        cols_m[i+2].metric(k, str(v))

    st.subheader("📊 Business Insights")
    for ins in st.session_state.business_insights:
        st.info(ins)

# ======================================================
# CHAT — always visible, works after analysis
# ======================================================
st.divider()
st.subheader("💬 Ask Questions About Your Data")

if not st.session_state.analyzed:
    st.info("⚠️ Run analysis first to enable chat")
else:
    # Ensure list
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    # Show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    user_input = st.chat_input("Ask about your business data...")

    if user_input:
        # Save user message
        st.session_state.chat_history.append({
            "role": "user", "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
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
                        "role": "assistant", "content": response
                    })
                except Exception as e:
                    st.error(f"Chat error: {e}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ======================================================
# TECHNICAL VIEW
# ======================================================
with st.expander("⚙️ Technical Details"):
    if st.session_state.analyzed:
        mc  = st.session_state.model_card
        raw = mc.get("metrics", {})
        cv  = raw.get("cv", {})
        ho  = raw.get("holdout", {})
        st.markdown("### 📈 Full Metrics")
        if st.session_state.problem_type == "regression":
            st.write(f"R²:      {round(ho.get('r2',0), 4)}")
            st.write(f"MAE:     {round(ho.get('mae',0), 2)}")
            st.write(f"RMSE:    {round(ho.get('rmse',0), 2)}")
        else:
            st.write(f"Accuracy: {round(ho.get('accuracy',0), 4)}")
            st.write(f"F1:       {round(ho.get('f1',0), 4)}")
        st.write(f"CV Mean:  {round(cv.get('mean',0), 4)}")
        st.write(f"CV Std:   {round(cv.get('std',0), 4)}")
    else:
        st.info("Run analysis to see metrics")
