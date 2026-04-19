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
    "pending_question": None,
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
    raw = pd.read_csv(uploaded)
    st.session_state.df = raw
    st.session_state.profile = basic_profile(raw)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

if st.sidebar.button("🎯 Load Demo"):
    demo = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
    st.session_state.df = demo
    st.session_state.profile = basic_profile(demo)
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
mode = st.selectbox(
    "Choose Analysis Type",
    ["📊 Revenue Analysis", "👤 Churn Analysis"]
)

try:
    if mode == "📊 Revenue Analysis":
        df = adaptive_preprocess(df, mode="revenue")
        target = "Revenue"
    else:
        df = adaptive_preprocess(df, mode="churn")
        target = "Churn"

    st.success("✅ Data processed")

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

        # 🔥 FIXED (returns metrics)
        results, best_model_name, metrics = train_models(X, y, problem)

        model = joblib.load("models/best_model.pkl")

        # SHAP (safe sample)
        try:
            sample_X = X.sample(min(1000, len(X)), random_state=42)
            explainer = shap.Explainer(model, sample_X)
            sv = explainer(sample_X)

            insights = generate_business_impact(
                sv.values, sample_X, problem, target
            )
        except:
            insights = ["SHAP unavailable"]

        st.session_state.update({
            "X": X,
            "y": y,
            "model": model,
            "problem_type": problem,
            "business_insights": insights,
            "analyzed": True,
            "model_card": {
                "model": best_model_name,
                "features": X.shape[1],
                "metrics": metrics
            }
        })

        st.success(f"✅ Model trained: {best_model_name}")

# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:

    st.subheader("🏆 Model Summary")
    st.write(st.session_state.model_card)

    st.subheader("📊 Key Drivers")
    for i in st.session_state.business_insights:
        st.info(i)

# ======================================================
# CHAT (NO RERUN BUG)
# ======================================================
st.divider()
st.subheader("💬 Ask DataAgentX")

if not st.session_state.analyzed:
    st.info("Run analysis first")
else:

    # history
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

    user_input = st.chat_input("Ask anything...")

    if user_input:
        st.session_state.pending_question = user_input

    if st.session_state.pending_question:

        q = st.session_state.pending_question

        with st.chat_message("user"):
            st.write(q)

        st.session_state.chat_history.append({"role": "user", "content": q})

        with st.chat_message("assistant"):
            try:
                resp = chat_with_data(
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
                st.write(resp)
            except Exception as e:
                resp = "Error"
                st.error(str(e))

        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.session_state.pending_question = None

# ======================================================
# TECHNICAL VIEW
# ======================================================

# ======================================================
# TECHNICAL VIEW
# ======================================================
with st.expander("⚙️ Technical Details"):

    st.markdown("### 🧠 Pipeline")
    st.code("""
Upload → Adaptive Preprocess → Feature Engineering → AutoML → CV → SHAP → Insights → Chat
""")

    st.markdown("### 🤖 Model Info")
    st.write(st.session_state.model_card)

    st.markdown("### 📈 Metrics")

    if st.session_state.model_card is not None:

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
