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
from src.adaptive_preprocess import adaptive_preprocess

# ======================================================
# CONFIG + UI
# ======================================================
st.set_page_config(page_title="DataAgentX", layout="wide", page_icon="🚀")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0b0f19; color:#e8ecf1; }
[data-testid="stSidebar"] { background:#0f1320; border-right:1px solid #1e2535; }
.block-container { padding-top: 1.5rem; }
.stButton>button { border-radius:10px; font-weight:600; }
[data-testid="stMetric"] {
  background:#141826; border:1px solid #1e2535; border-radius:12px; padding:16px !important;
}
</style>
""", unsafe_allow_html=True)

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
    "agent_narrative": None, "shap_top_features": None,
    "pending_question": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown("## 🚀 DataAgentX")
st.sidebar.caption("AI Business Intelligence")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
    st.session_state.df = raw
    st.session_state.profile = basic_profile(raw)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

# Demo
if st.sidebar.button("🎯 Load Demo"):
    demo = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
    st.session_state.df = demo
    st.session_state.profile = basic_profile(demo)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

df = st.session_state.df
profile = st.session_state.profile

if df is None:
    st.info("👈 Upload a dataset or use demo")
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

    with st.spinner("Training..."):

        X, y = prepare_features(df, profile, target)
        X = X.select_dtypes(include=np.number).fillna(0)

        problem = detect_problem_type(y)
        results, best_model = train_models(X, y, problem)

        model = joblib.load("models/best_model.pkl")

        # SHAP
        try:
            explainer = shap.Explainer(model, X)
            sv = explainer(X)
            insights = generate_business_impact(sv.values, X, problem, target)
        except:
            insights = ["SHAP unavailable"]

        st.session_state.update({
            "X": X, "y": y, "model": model,
            "problem_type": problem,
            "business_insights": insights,
            "analyzed": True,
            "model_card": {
                "model": best_model,
                "features": X.shape[1]
            }
        })

        st.success(f"✅ Model trained: {best_model}")

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
# CHAT (FIXED UX)
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
    c1, c2, c3 = st.columns(3)

    if c1.button("📉 Why drop?"):
        st.session_state.pending_question = f"Why did {target} decrease?"

    if c2.button("📊 Drivers?"):
        st.session_state.pending_question = f"What drives {target}?"

    if c3.button("📈 Improve?"):
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
# # 📊 TECHNICAL VIEW (HIDDEN)
# # ======================================================
# with st.expander("⚙️ Technical Details (for advanced users)"):
#     st.markdown("### Pipeline")
#     st.code("""
# Upload → Preprocess → Feature Engineering → AutoML → SHAP → Insights → Chat
# """)

#     st.markdown("### Model Info")
#     st.write(st.session_state.model_card)

#     if st.session_state.X is not None:
#         st.markdown("### Features Used")
#         st.write(st.session_state.X.columns.tolist())

#     st.markdown("### Problem Type")
#     st.write(st.session_state.problem_type)

# ======================================================
# 📊 TECHNICAL VIEW (ADVANCED)
# ======================================================
with st.expander("⚙️ Technical Details (for advanced users)"):

    st.markdown("### 🧠 Pipeline")
    st.code("""
Upload → Adaptive Preprocess → Feature Engineering → AutoML → Cross Validation → SHAP → Insights → Chat
""")

    # =========================
    # MODEL INFO
    # =========================
    st.markdown("### 🤖 Model Info")
    st.write(st.session_state.model_card)

    # =========================
    # PROBLEM TYPE
    # =========================
    st.markdown("### 🧪 Problem Type")
    st.write(st.session_state.problem_type)

    # =========================
    # FEATURES
    # =========================
    if st.session_state.X is not None:
        st.markdown("### 📦 Features Used")
        st.write(st.session_state.X.columns.tolist())

    # =========================
    # METRICS
    # =========================
    st.markdown("### 📈 Performance Metrics")

    try:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from sklearn.metrics import accuracy_score, f1_score

        model = st.session_state.model
        X = st.session_state.X
        y = st.session_state.y
        problem = st.session_state.problem_type

        preds = model.predict(X)

        if problem == "regression":

            r2 = r2_score(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            mae = mean_absolute_error(y, preds)

            st.write("R²:", round(r2, 4))
            st.write("RMSE:", round(rmse, 2))
            st.write("MAE:", round(mae, 2))

            # Cross Validation
            from sklearn.model_selection import KFold, cross_val_score

            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

            st.write("CV Mean:", round(cv_scores.mean(), 4))
            st.write("CV Std:", round(cv_scores.std(), 4))

        else:

            acc = accuracy_score(y, preds)
            f1 = f1_score(y, preds)

            st.write("Accuracy:", round(acc, 4))
            st.write("F1 Score:", round(f1, 4))

    except Exception as e:
        st.warning(f"Metrics unavailable: {e}")

    # =========================
    # SHAP (ADVANCED)
    # =========================
    st.markdown("### 🔍 SHAP Explainability")

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        st.write("Top SHAP Features:")

        mean_shap = np.abs(shap_values.values).mean(axis=0)
        shap_importance = pd.Series(mean_shap, index=X.columns).sort_values(ascending=False)

        st.write(shap_importance.head(10))

        # Optional plot
        st.markdown("#### SHAP Summary Plot")
        st.pyplot(shap.summary_plot(shap_values, X, show=False))

    except Exception as e:
        st.warning(f"SHAP failed: {e}")



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
