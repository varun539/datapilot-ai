import streamlit as st
import pandas as pd
import numpy as np
import shap
import os

from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data, generate_agent_narrative
from src.eda import basic_profile
from src.adaptive_preprocess import adaptive_preprocess

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataAgentX", layout="wide")
api_key = os.getenv("OPENAI_API_KEY")

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "df": None,
    "profile": None,
    "analyzed": False,
    "chat_history": [],
    "model": None,
    "model_card": None,
    "business_insights": [],
    "processed_df": None,
    "problem_type": None,
    "target": None,
    "agent_narrative": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# UPLOAD
# ======================================================
st.sidebar.title("🚀 DataAgentX")

file = st.sidebar.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    st.session_state.df = df
    st.session_state.profile = basic_profile(df)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

df = st.session_state.df

if df is None:
    st.info("Upload dataset to start")
    st.stop()

# ======================================================
# MODE
# ======================================================
st.subheader("Choose Analysis Type")

mode = st.selectbox(
    "Mode",
    ["Revenue Analysis", "Custom"]
)

# ======================================================
# PREPROCESS
# ======================================================
if mode == "Revenue Analysis":
    try:
        processed_df = adaptive_preprocess(df, "revenue")
        target = "Revenue"
    except:
        processed_df = df.copy()
        target = df.select_dtypes(include=np.number).columns[0]
else:
    processed_df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Target", numeric_cols)

st.session_state.target = target

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Analyze"):

    try:
        X, y = prepare_features(processed_df, {}, target)
        X = X.select_dtypes(include=np.number).fillna(0)

        problem = detect_problem_type(y)

        # IMPORTANT: use model directly (no joblib load)
        results, best_model, metrics = train_models(X, y, problem)
        model = best_model

        # SHAP
        try:
            sample_X = X.sample(min(300, len(X)))
            explainer = shap.Explainer(model, sample_X)
            sv = explainer(sample_X)

            insights = generate_business_impact(
                sv.values, sample_X, problem, target
            )
        except:
            insights = ["Basic insights available"]

        st.session_state.update({
            "model": model,
            "model_card": {
                "model": type(model).__name__,
                "metrics": metrics
            },
            "business_insights": insights,
            "processed_df": processed_df,
            "problem_type": problem,
            "analyzed": True,
            "agent_narrative": None
        })

        st.success("Analysis complete")

    except Exception as e:
        st.error(f"Analysis failed: {e}")

# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:

    st.subheader("Insights")

    for i in st.session_state.business_insights:
        st.info(i)

    # ======================================================
    # ALERTS
    # ======================================================
    st.subheader("Alerts")

    df_disp = st.session_state.processed_df if st.session_state.processed_df is not None else df

    try:
        vals = df_disp[target].dropna()
        if len(vals) > 10:
            if vals.tail(5).mean() < vals.mean():
                st.warning("Revenue trending down")
    except:
        pass

    # ======================================================
    # EXECUTIVE SUMMARY
    # ======================================================
    if api_key:

        if st.session_state.agent_narrative is None:
            with st.spinner("Generating summary..."):
                try:
                    st.session_state.agent_narrative = generate_agent_narrative(
                        api_key,
                        st.session_state.model_card,
                        st.session_state.business_insights,
                        {},
                        [],
                        st.session_state.problem_type,
                        target
                    )
                except:
                    st.session_state.agent_narrative = "AI unavailable"

        st.subheader("Executive Analysis")
        st.write(st.session_state.agent_narrative)

# ======================================================
# CHAT (FINAL STABLE)
# ======================================================
st.subheader("Ask AI")

if st.session_state.analyzed:

    # show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask something")

    if user_input:

        # save user
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                response = chat_with_data(
                    api_key,
                    user_input,
                    st.session_state.chat_history[:-1],  # FIX
                    st.session_state.model_card,
                    {},
                    st.session_state.processed_df,
                    st.session_state.problem_type,
                    target,
                    st.session_state.business_insights
                )

                st.write(response)

        # save AI
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

# ======================================================
# TECHNICAL
# ======================================================
with st.expander("Technical Details"):
    st.write(st.session_state.model_card)
