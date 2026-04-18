import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

# =========================
# IMPORTS
# =========================
from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data, suggest_target_column, generate_agent_narrative
from src.report import generate_pdf_report
from src.eda import basic_profile

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="DataPilot AI", layout="wide")

# =========================
# API KEY (FIX 🔥)
# =========================
api_key = os.getenv("OPENAI_API_KEY")

# =========================
# SESSION STATE
# =========================
for key in [
    "X", "y", "model", "problem_type",
    "target_col", "business_insights", "chat_history"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================
# FILE UPLOAD
# =========================
st.sidebar.header("📂 Upload Data")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    profile = basic_profile(df)
else:
    df = None

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🚀 DataPilot AI")

dev_mode = st.sidebar.toggle("🧠 Advanced Mode")

pages = ["🏠 Home", "📊 Dashboard", "💬 Chat", "📄 Report"]

if dev_mode:
    pages += ["🧠 Advanced", "⚙️ Technical"]

page = st.sidebar.radio("Navigate", pages)

# ======================================================
# 🏠 HOME
# ======================================================
if page == "🏠 Home":

    st.title("🚀 DataPilot AI")
    st.markdown("### Turn your business data into insights in seconds")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Insights", "Instant")
    col2.metric("🤖 AI Powered", "Yes")
    col3.metric("📄 Reports", "1 Click")

    st.info("👉 Upload dataset → Go to Dashboard")

# ======================================================
# 📊 DASHBOARD
# ======================================================
elif page == "📊 Dashboard":

    if df is None:
        st.warning("Upload dataset first")
        st.stop()

    st.header("📊 Business Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing", int(df.isnull().sum().sum()))

    st.divider()

    # =========================
    # AUTO TARGET
    # =========================
    target = suggest_target_column(
        api_key,
        df.columns.tolist(),
        df
    )

    st.info(f"🎯 AI selected target: {target}")

    # =========================
    # ANALYSIS
    # =========================
    if st.button("🚀 Analyze My Business", key="analyze_btn"):

        with st.spinner("Analyzing..."):

            X = prepare_features(df, profile, target, training=True)
            y = pd.to_numeric(df[target], errors="coerce").fillna(df[target].median())

            problem = detect_problem_type(y)
            results, best_model_name = train_models(X, y, problem)

            model = joblib.load("models/best_model.pkl")

            st.session_state.update({
                "X": X,
                "y": y,
                "model": model,
                "problem_type": problem,
                "target_col": target
            })

            st.success(f"🏆 Best Model: {best_model_name}")
            st.dataframe(results, use_container_width=True)

            # =========================
            # SHAP INSIGHTS
            # =========================
            try:
                explainer = shap.Explainer(model, X)
                shap_vals = explainer(X)

                insights = generate_business_impact(
                    shap_vals.values, X, problem, target
                )

                st.subheader("📊 Key Drivers")

                for i in insights:
                    st.info(i)

                st.session_state.business_insights = insights

            except Exception:
                st.warning("SHAP not available")

            # =========================
            # AI SUMMARY (FIXED)
            # =========================
            st.subheader("🤖 AI Summary")

            try:
                if api_key:
                    summary = generate_agent_narrative(
                        api_key,
                        df,
                        results,
                        target,
                        st.session_state.get("business_insights", [])
                    )
                    st.success(summary)
                else:
                    st.info("Add OPENAI_API_KEY to enable AI summary")

            except Exception:
                st.warning("Could not generate AI summary")

# ======================================================
# 💬 CHAT
# ======================================================
elif page == "💬 Chat":

    if df is None:
        st.warning("Upload dataset first")
        st.stop()

    st.subheader("💬 Ask Your Data")

    question = st.text_input("Ask something about your data")

    if st.button("Ask", key="ask_btn"):

        try:
            response = chat_with_data(
                api_key,
                question,
                st.session_state.get("chat_history", []),
                {},
                {},
                df,
                st.session_state.get("problem_type"),
                st.session_state.get("target_col"),
                st.session_state.get("business_insights", [])
            )

            st.write(response)

        except Exception:
            st.warning("Chat not available (check API key)")

# ======================================================
# 📄 REPORT
# ======================================================
elif page == "📄 Report":

    st.header("📄 Business Report")

    if st.session_state.get("business_insights"):

        if st.button("📄 Generate Report"):

            path = generate_pdf_report(
                {},
                st.session_state.get("business_insights", [])
            )

            with open(path, "rb") as f:
                st.download_button("⬇️ Download Report", f, "report.pdf")

    else:
        st.warning("Run analysis first")

# ======================================================
# 🧠 ADVANCED
# ======================================================
elif page == "🧠 Advanced":
    st.subheader("Advanced features (hidden)")

# ======================================================
# ⚙️ TECHNICAL VIEW
# ======================================================
elif page == "⚙️ Technical":

    st.header("⚙️ System Architecture")

    st.code("""
User → Upload Data
     ↓
Feature Engineering
     ↓
AutoML
     ↓
Best Model
     ↓
SHAP Explainability
     ↓
Business Insights
     ↓
AI Chat Layer
""")

    st.subheader("Model Info")
    st.write(type(st.session_state.get("model")))

    st.subheader("Features")
    if st.session_state.get("X") is not None:
        st.write(st.session_state["X"].columns.tolist())

    st.subheader("Target")
    st.write(st.session_state.get("target_col"))

    st.subheader("Problem Type")
    st.write(st.session_state.get("problem_type"))

    st.subheader("Insights")
    for i in st.session_state.get("business_insights", []) or []:
        st.write("-", i)
