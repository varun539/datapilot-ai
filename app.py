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
from src.agent import chat_with_data, generate_agent_narrative
from src.report import generate_pdf_report
from src.eda import basic_profile

# 🔥 NEW (adaptive)
from adaptive_preprocess import adaptive_preprocess

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataAgentX", layout="wide", page_icon="🚀")

api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

# ======================================================
# SESSION STATE
# ======================================================
DEFAULTS = {
    "df": None, "profile": None, "X": None, "y": None,
    "model": None, "problem_type": None, "target_col": None,
    "business_insights": None, "analyzed": False,
    "chat_history": [], "model_card": None,
    "agent_narrative": None, "shap_top_features": None,
    "feature_schema": None,
    "pending_question": None
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🚀 DataAgentX")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.session_state.df = df_raw
    st.session_state.profile = basic_profile(df_raw)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

df = st.session_state.df
profile = st.session_state.profile

if df is None:
    st.info("👈 Upload a dataset")
    st.stop()

# ======================================================
# 🧠 MODE SELECTOR (NEW)
# ======================================================
mode = st.selectbox(
    "Choose Analysis Type",
    ["📊 Revenue Analysis", "👤 Churn Analysis"]
)

# ======================================================
# 🔥 ADAPTIVE PREPROCESS (CRITICAL)
# ======================================================
try:
    if mode == "📊 Revenue Analysis":
        df = adaptive_preprocess(df, mode="revenue")
        target = "Revenue"
    else:
        df = adaptive_preprocess(df, mode="churn")
        target = "Churn"

    st.success("✅ Data auto-processed")

except Exception as e:
    st.error(f"Preprocess failed: {e}")
    st.stop()

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Analyze"):

    with st.spinner("Training model..."):

        X, y = prepare_features(df, profile, target)

        X = X.select_dtypes(include=np.number).fillna(0)

        problem = detect_problem_type(y)
        results, best_model_name = train_models(X, y, problem)

        model = joblib.load("models/best_model.pkl")

        # SHAP
        try:
            explainer = shap.Explainer(model, X)
            shap_vals = explainer(X)

            insights = generate_business_impact(
                shap_vals.values, X, problem, target
            )
        except:
            insights = ["SHAP failed"]

        st.session_state.update({
            "X": X,
            "y": y,
            "model": model,
            "problem_type": problem,
            "business_insights": insights,
            "analyzed": True,
            "model_card": {
                "model": best_model_name,
                "features": X.shape[1]
            }
        })

# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:

    st.subheader("🏆 Model Performance")
    st.write(st.session_state.model_card)

    st.subheader("📊 Key Drivers")
    for i in st.session_state.business_insights:
        st.info(i)

# ======================================================
# 💬 CHAT (NO REFRESH FIXED)
# ======================================================
st.divider()
st.subheader("💬 Ask DataAgentX")

if not st.session_state.analyzed:
    st.info("Run analysis first")
else:

    # show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # quick buttons
    col1, col2, col3 = st.columns(3)

    if col1.button("📉 Why drop?"):
        st.session_state.pending_question = f"Why did {target} decrease?"

    if col2.button("📊 Drivers?"):
        st.session_state.pending_question = f"What drives {target}?"

    if col3.button("📈 Improve?"):
        st.session_state.pending_question = f"How to improve {target}?"

    user_input = st.chat_input("Ask your own question")

    if user_input:
        st.session_state.pending_question = user_input

    # 🔥 PROCESS ONCE
    if st.session_state.pending_question is not None:

        q = st.session_state.pending_question

        with st.chat_message("user"):
            st.write(q)

        st.session_state.chat_history.append({
            "role": "user",
            "content": q
        })

        with st.chat_message("assistant"):

            try:
                response = chat_with_data(
                    api_key,
                    q,
                    st.session_state.chat_history.copy(),
                    st.session_state.model_card,
                    profile,
                    df,
                    st.session_state.problem_type,
                    target,
                    st.session_state.business_insights
                )

                st.write(response)

            except Exception as e:
                st.error(str(e))
                response = "Error"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

        st.session_state.pending_question = None

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ======================================================
# REPORT
# ======================================================
if st.session_state.analyzed:

    if st.button("📄 Download Report"):

        path = generate_pdf_report(
            st.session_state.model_card,
            st.session_state.business_insights
        )

        with open(path, "rb") as f:
            st.download_button("Download PDF", f)
