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
from src.agent import chat_with_data, suggest_target_column, generate_agent_narrative
from src.eda import basic_profile

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataAgentX", layout="wide")

api_key = os.getenv("OPENAI_API_KEY")

# ======================================================
# SESSION STATE
# ======================================================
keys = [
    "df", "profile", "X", "y", "model",
    "problem_type", "target_col",
    "business_insights", "analyzed",
    "chat_history", "model_card",
    "agent_narrative", "df_used"
]

for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

if st.session_state.chat_history is None:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ======================================================
# UPLOAD
# ======================================================
st.title("🚀 DataAgentX")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    st.session_state.df = df
    st.session_state.profile = basic_profile(df)
    st.session_state.analyzed = False
    st.session_state.chat_history = []
    st.session_state.target_col = None

df = st.session_state.df
profile = st.session_state.profile

if df is None:
    st.stop()

# ======================================================
# TARGET
# ======================================================
if st.session_state.target_col is None:
    st.session_state.target_col = suggest_target_column(
        api_key, df.columns.tolist(), df
    )

target = st.selectbox(
    "🎯 Target",
    df.columns,
    index=df.columns.get_loc(st.session_state.target_col)
)

st.session_state.target_col = target

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Analyze"):

    with st.spinner("Training model..."):

        X, y = prepare_features(df, profile, target)

        X = X.select_dtypes(include=np.number).fillna(0)

        problem = detect_problem_type(y)
        results, best_model = train_models(X, y, problem)

        model = joblib.load("models/best_model.pkl")

        # SHAP
        try:
            explainer = shap.Explainer(model, X)
            shap_vals = explainer(X)

            insights = generate_business_impact(
                shap_vals.values, X, problem, target
            )
        except:
            insights = ["SHAP not available"]

        st.session_state.update({
            "X": X,
            "y": y,
            "model": model,
            "problem_type": problem,
            "business_insights": insights,
            "analyzed": True,
            "model_card": {
                "model": best_model,
                "features": X.shape[1]
            },
            "df_used": df.copy()
        })

# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:

    st.subheader("🏆 Model Results")
    st.dataframe(st.session_state.model_card)

    st.subheader("📊 Key Drivers")
    for i in st.session_state.business_insights:
        st.info(i)

# ======================================================
# CHAT (FIXED)
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

    # PROCESS ONLY ONCE
    if st.session_state.pending_question is not None:

        q = st.session_state.pending_question

        # show user
        with st.chat_message("user"):
            st.write(q)

        st.session_state.chat_history.append({
            "role": "user",
            "content": q
        })

        # assistant
        with st.chat_message("assistant"):

            placeholder = st.empty()
            full = ""

            try:
                response = chat_with_data(
                    api_key,
                    q,
                    st.session_state.chat_history.copy(),  # IMPORTANT FIX
                    st.session_state.model_card,
                    profile,
                    st.session_state.df_used,  # FIXED
                    st.session_state.problem_type,
                    target,
                    st.session_state.business_insights
                )

                # streaming effect
                for w in response.split():
                    full += w + " "
                    placeholder.markdown(full + "▌")

                placeholder.markdown(full)

            except Exception as e:
                placeholder.error(str(e))
                full = "Error"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full
        })

        # RESET TRIGGER
        st.session_state.pending_question = None

    # clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.pending_question = None
        st.rerun()
