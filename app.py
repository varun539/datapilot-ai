

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# Internal modules
from src.pipeline import prepare_features
from src.data_loader import load_csv
from src.eda import basic_profile, plot_numeric_distributions, plot_correlation_heatmap
from src.automl import detect_problem_type, train_models, detect_training_mode, detect_data_leakage
from src.data_quality import calculate_data_quality
from src.model_registry import register_model, get_all_models
from src.impact import generate_business_impact
from src.report import generate_pdf_report
from src.experiments import log_experiment, load_experiments
from src.agent import generate_agent_narrative, chat_with_data, suggest_target_column

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataPilot AI", layout="wide", page_icon="🚀")
api_key = st.secrets.get("OPENAI_API_KEY", None)

# ======================================================
# CACHE
# ======================================================
@st.cache_data
def load_cached_csv(file):
    return load_csv(file)

# ======================================================
# SESSION STATE
# ======================================================
STATE_KEYS = [
    "X","y","model","problem_type","target_col",
    "training_mode","feature_schema",
    "model_card","business_insights","residual_std",
    "agent_narrative","chat_history","shap_top_features"
]

for k in STATE_KEYS:
    st.session_state.setdefault(k, None)

st.session_state.setdefault("chat_history", [])

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🚀 DataPilot AI")
st.sidebar.caption("Agentic AutoML Platform")

if api_key:
    st.sidebar.success("OpenAI Connected")
else:
    st.sidebar.warning("No API key")

page = st.sidebar.radio("Navigate", [
    "📊 Data Overview",
    "📈 Visual Analytics",
    "🤖 AutoML",
    "🧠 Explainability",
    "💬 Chat",
    "🔮 Prediction",
    "🧪 Experiments",
    "📦 Models",
    "⬇️ Downloads"
])

# ======================================================
# DATA LOAD
# ======================================================
st.title("🚀 DataPilot AI")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload dataset to begin")
    st.stop()

df = load_cached_csv(uploaded_file)
profile = basic_profile(df)

st.sidebar.metric("Rows", df.shape[0])
st.sidebar.metric("Columns", df.shape[1])

# ======================================================
# 📊 DATA OVERVIEW
# # ======================================================
# if page == "📊 Data Overview":

#     score, level, messages = calculate_data_quality(profile)

#     st.metric("Quality Score", f"{score}/100")
#     st.write(level)

#     for m in messages:
#         st.warning(m)

#     st.dataframe(df.head())

#     # ✅ SAFE CORRELATION
#     numeric_cols = df.select_dtypes(include=np.number).columns

#     if len(numeric_cols) > 1:
#         corr = df[numeric_cols].corr().abs()
#         corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0)

#         if corr.shape[0] > 1:
#             try:
#                 np.fill_diagonal(corr.values, 0)
#             except:
#                 pass

#             top = corr.unstack().sort_values(ascending=False).drop_duplicates()

#             if not top.empty:
#                 f1, f2 = top.index[0]
#                 st.info(f"Strongest relation: {f1} ↔ {f2} ({top.iloc[0]:.2f})")
#         else:
#             st.info("Not enough numeric features for correlation.")
#     else:
#         st.info("Not enough numeric columns.")

#     # Missing values
#     missing = df.isnull().sum().sum()
#     st.warning(f"{missing} missing values") if missing else st.success("No missing values")

#     # AI target suggestion
#     if api_key and st.button("Suggest Target"):
#         target = suggest_target_column(api_key, df.columns.tolist(), df)
#         st.success(f"Suggested: {target}")

# ======================================================
# 📊 DATA OVERVIEW
# ======================================================
  if page == "📊 Data Overview":

    score, level, messages = calculate_data_quality(profile)

    st.metric("Quality Score", f"{score}/100")
    st.write(level)

    for m in messages:
        st.warning(m)

    st.dataframe(df.head())

    # =============================
    # ✅ SAFE CORRELATION
    # =============================
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation")

    else:
        corr = df[numeric_cols].corr()

        # Clean correlation matrix
        corr = corr.replace([np.inf, -np.inf], np.nan)
        corr = corr.dropna(how="all", axis=0).dropna(how="all", axis=1)

        if corr.shape[0] < 2:
            st.info("Not enough valid numeric features for correlation")

        else:
            np.fill_diagonal(corr.values, 0)

            # Get strongest pair
            top = corr.abs().unstack().sort_values(ascending=False)

            found = False

            for (f1, f2), val in top.items():
                if f1 != f2:
                    st.info(f"Strongest relation: {f1} ↔ {f2} ({val:.2f})")
                    found = True
                    break

            if not found:
                st.info("No strong correlations found")

    # =============================
    # ✅ MISSING VALUES
    # =============================
    missing = df.isnull().sum().sum()

    if missing:
        st.warning(f"{missing} missing values")
    else:
        st.success("No missing values")

    # =============================
    # 🤖 AI TARGET SUGGESTION
    # =============================
    if api_key and st.button("Suggest Target", key="suggest_target_btn"):
        target = suggest_target_column(api_key, df.columns.tolist(), df)
        st.success(f"Suggested: {target}")  
    
    
    

# ======================================================
# 📈 VISUALS
# ======================================================
elif page == "📈 Visual Analytics":

    for fig in plot_numeric_distributions(df, profile["numeric_cols"]):
        st.pyplot(fig)

    if len(profile["numeric_cols"]) >= 2:
        st.pyplot(plot_correlation_heatmap(df, profile["numeric_cols"]))

# ======================================================
# 🤖 AUTOML
# ======================================================
# ======================================================
# 🔥 AUTO TARGET SELECTION (REAL AUT0ML STYLE)
# ======================================================
# ======================================================
# 🤖 AUTOML
# # ======================================================


elif page == "🤖 AutoML":

    st.header("🤖 Automated Machine Learning")

    numeric_targets = df.select_dtypes(include="number").columns.tolist()
    target = st.selectbox("🎯 Select Target Column", numeric_targets)
    st.session_state.target_col = target

    # ✅ SINGLE BUTTON (FIXED)
    if st.button("🚀 Train Models", key="train_btn"):

        with st.spinner("Training models..."):

            X = prepare_features(df, profile, target, training=True)
            y = pd.to_numeric(df[target], errors="coerce").fillna(df[target].median())

            leak, _ = detect_data_leakage(X, y)
            if leak:
                st.error("⚠️ Data leakage detected")
                st.stop()

            problem = detect_problem_type(y)

            results, best_model_name = train_models(X, y, problem)
            model = joblib.load("models/best_model.pkl")

            st.session_state.update({
                "X": X,
                "y": y,
                "model": model,
                "problem_type": problem,
                "feature_schema": X.columns.tolist()
            })

            # ✅ CLEAN CV (NO CONFUSION)
            if problem == "regression":
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                score = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                score = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()

            st.success(f"🏆 Best Model: {best_model_name}")
            st.metric("CV Score", f"{score:.4f}")

            # ✅ SHAP FIXED
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)

                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]

                insights = generate_business_impact(
                    shap_vals, X, y, problem, target   # ✅ FIXED
                )

                st.session_state.business_insights = insights

                st.subheader("💼 Business Impact")
                for i in insights:
                    st.info(i)

            except:
                st.warning("SHAP not supported")

            st.dataframe(results, use_container_width=True)

            # ✅ MODEL CARD FIX (FOR CHAT)
            st.session_state.model_card = {
                "model": best_model_name,
                "problem": problem,
                "rows": df.shape[0],
                "features": X.shape[1],
                "target": target,
                "performance": {"CV Score": round(score, 4)}
            }







# elif page == "🤖 AutoML":

#     # 🔥 AUTO TARGET SELECTION (CLEAN)
#     numeric_targets = []

#     for c in df.columns:
#         try:
#             pd.to_numeric(df[c])
#             numeric_targets.append(c)
#         except:
#             continue

#     # remove obvious useless columns
#     filtered_targets = []
#     for c in numeric_targets:
#         col_lower = c.lower()

#         if any(k in col_lower for k in ["id", "index", "code", "number"]):
#             continue

#         filtered_targets.append(c)

# #     if not filtered_targets:
# #         filtered_targets = numeric_targets

# #     target = st.selectbox("🎯 Select Target Column", filtered_targets)
# #     st.session_state.target_col = target

##1111111111111111111111111111111111111111111111111
# ======================================================
# 🤖 AUTOML
# ======================================================
elif page == "🤖 AutoML":

    st.header("🤖 Automated Machine Learning")

    # ✅ SIMPLE TARGET SELECTION
    numeric_targets = df.select_dtypes(include="number").columns.tolist()
    target = st.selectbox("🎯 Select Target Column", numeric_targets)
    st.session_state.target_col = target

    # ✅ SINGLE BUTTON (NO DUPLICATE ERROR)
    if st.button("🚀 Train Models", key="train_btn"):

        with st.spinner("Training models..."):

            # =============================
            # DATA PREP
            # =============================
            X = prepare_features(df, profile, target, training=True)
            y = pd.to_numeric(df[target], errors="coerce").fillna(df[target].median())

            # =============================
            # LEAKAGE CHECK
            # =============================
            leak, _ = detect_data_leakage(X, y)
            if leak:
                st.error("⚠️ Data leakage detected")
                st.stop()

            # =============================
            # TRAIN
            # =============================
            problem = detect_problem_type(y)

            results, best_model_name = train_models(X, y, problem)
            model = joblib.load("models/best_model.pkl")

            st.session_state.update({
                "X": X,
                "y": y,
                "model": model,
                "problem_type": problem,
                "feature_schema": X.columns.tolist()
            })

            # =============================
            # CROSS VALIDATION (STABLE)
            # =============================
            from sklearn.model_selection import KFold, StratifiedKFold

            if problem == "regression":
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                score = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                score = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()

            st.success(f"🏆 Best Model: {best_model_name}")
            st.metric("CV Score", f"{score:.4f}")

            # =============================
            # SHAP + BUSINESS IMPACT (FIXED)
            # =============================
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)

                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]

                insights = generate_business_impact(
                    shap_vals, X, y, problem, target   # ✅ FIXED
                )

                st.session_state.business_insights = insights

                st.subheader("💼 Business Impact")
                for i in insights:
                    st.info(i)

            except:
                st.warning("SHAP not supported")

            # =============================
            # RESULTS TABLE
            # =============================
            st.dataframe(results, use_container_width=True)

            # =============================
            # MODEL CARD (FOR CHAT FIX)
            # =============================
            st.session_state.model_card = {
                "model": best_model_name,
                "problem": problem,
                "rows": df.shape[0],
                "features": X.shape[1],
                "target": target,
                "performance": {"CV Score": round(score, 4)}
            }













###111111111111111111111111111111111111111111111111
# ======================================================
# 🧠 EXPLAINABILITY
# ======================================================
elif page == "🧠 Explainability":

    if st.session_state.model is None:
        st.warning("Train first")
        st.stop()

    try:
        Xs = st.session_state.X.sample(min(200, len(st.session_state.X)))
        explainer = shap.TreeExplainer(st.session_state.model)
        shap_vals = explainer.shap_values(Xs)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        fig = plt.figure()
        shap.summary_plot(shap_vals, Xs, show=False)
        st.pyplot(fig)

    except:
        st.error("SHAP failed")

# ======================================================
# 💬 CHAT
# ======================================================
# elif page == "💬 Chat":

#     if not api_key:
#         st.warning("API key required")
#         st.stop()

#     user_input = st.chat_input("Ask anything...")

#     if user_input:
#         response = chat_with_data(
#             api_key,
#             user_input,
#             st.session_state.chat_history,
#             None,
#             profile,
#             df,
#             st.session_state.problem_type,
#             st.session_state.target_col,
#             []
#         )
#         st.write(response)

elif page == "💬 Chat":

    if not api_key:
        st.warning("API key required")
        st.stop()

    model_card = st.session_state.model_card or {}

    user_input = st.chat_input("Ask anything...")

    if user_input:
        response = chat_with_data(
            api_key,
            user_input,
            st.session_state.chat_history,
            model_card,   # ✅ FIXED
            profile,
            df,
            st.session_state.problem_type,
            st.session_state.target_col,
            st.session_state.business_insights or []
        )

        st.write(response)






# ======================================================
# 🔮 PREDICTION
# ======================================================
elif page == "🔮 Prediction":

    if st.session_state.model is None:
        st.warning("Train first")
        st.stop()

    schema = st.session_state.feature_schema
    inputs = {col: st.number_input(col, 0.0) for col in schema}

    if st.button("Predict"):
        Xp = pd.DataFrame([inputs])
        pred = st.session_state.model.predict(Xp)[0]
        st.success(f"Prediction: {pred}")

# ======================================================
# 🧪 EXPERIMENTS
# ======================================================
elif page == "🧪 Experiments":
    exp = load_experiments()
    st.dataframe(pd.DataFrame(exp)) if exp else st.info("No experiments")

# ======================================================
# 📦 MODELS
# ======================================================
elif page == "📦 Models":
    st.dataframe(pd.DataFrame(get_all_models()))

# ======================================================
# ⬇️ DOWNLOADS
# ======================================================
elif page == "⬇️ Downloads":

    if os.path.exists("models/best_model.pkl"):
        with open("models/best_model.pkl", "rb") as f:
            st.download_button("Download Model", f, "model.pkl")














