import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from src.pipeline import prepare_features
from src.data_loader import load_csv
from src.eda import basic_profile, plot_numeric_distributions, plot_correlation_heatmap
from src.automl import (
    detect_problem_type,
    train_models,
    detect_class_imbalance,
    detect_training_mode,
    detect_data_leakage
)
from src.data_quality import calculate_data_quality
from src.model_registry import register_model, get_all_models

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Varun's DataPilot AI", layout="wide")

# ======================================================
# CACHE
# ======================================================
@st.cache_data
def load_cached_csv(file):
    return load_csv(file)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🚀 Varun's DataPilot AI")
st.sidebar.caption("Production AutoML Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Data Overview",
        "📈 Visual Analytics",
        "🤖 AutoML",
        "🧠 Explainability",
        "🔮 Prediction",
        "📦 Model Registry",
        "⬇️ Downloads"
    ]
)

# ======================================================
# SESSION STATE INIT
# ======================================================
for key in [
    "X", "y", "model", "problem_type", "target_col",
    "training_mode", "feature_schema", "model_card",
    "residual_std"
]:
    st.session_state.setdefault(key, None)

st.session_state.setdefault("handle_imbalance", True)

# ======================================================
# HEADER
# ======================================================
st.title("🚀 Varun's DataPilot AI")
st.caption("End-to-End AutoML Platform")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin")
    st.stop()

df = load_cached_csv(uploaded_file)
profile = basic_profile(df)

st.sidebar.success("Dataset Loaded")
st.sidebar.metric("Rows", df.shape[0])
st.sidebar.metric("Columns", df.shape[1])

# ======================================================
# DATA OVERVIEW
# ======================================================
if page == "📊 Data Overview":
    score, level, messages = calculate_data_quality(profile)
    st.metric("Quality Score", f"{score}/100")
    st.markdown(f"### {level}")
    for msg in messages:
        st.warning(msg)
    st.dataframe(df.head(), use_container_width=True)

# ======================================================
# VISUAL ANALYTICS
# ======================================================
elif page == "📈 Visual Analytics":
    for fig in plot_numeric_distributions(df, profile["numeric_cols"]):
        st.pyplot(fig, use_container_width=True)
    heatmap = plot_correlation_heatmap(df, profile["numeric_cols"])
    if heatmap:
        st.pyplot(heatmap, use_container_width=True)

# ======================================================
# AUTOML
# ======================================================
elif page == "🤖 AutoML":
    st.header("🤖 Automated Machine Learning")

    candidate_targets = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col])
            candidate_targets.append(col)
        except:
            pass

    target_col = st.selectbox("🎯 Select Target Column", candidate_targets)
    st.session_state.target_col = target_col

    st.session_state.handle_imbalance = st.checkbox(
        "Handle Class Imbalance Automatically", value=True
    )

    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):

            X = prepare_features(df, profile, target_col, training=True)
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(df[target_col].median())

            # 🔒 Data Leakage Guard
            is_leakage, leaks = detect_data_leakage(X, y)
            if is_leakage:
                st.error("⚠️ Data Leakage Detected. Training stopped.")
                for f, c in leaks:
                    st.write(f"{f}: corr={c:.3f}")
                st.stop()

            problem_type = detect_problem_type(y)
            training_mode = detect_training_mode(df, target_col, profile)

            results, best_model_name = train_models(
                X, y, problem_type, st.session_state.handle_imbalance
            )

            model = joblib.load("models/best_model.pkl")

            # Store everything
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.model = model
            st.session_state.problem_type = problem_type
            st.session_state.training_mode = training_mode
            st.session_state.feature_schema = X.columns.tolist()

            # Residuals for confidence
            if problem_type == "regression":
                preds = model.predict(X)
                st.session_state.residual_std = np.std(y - preds)

            # 📦 Model Card
            if problem_type == "regression":
                r2 = np.corrcoef(y, preds)[0, 1] ** 2
                perf = {"R2": round(r2, 4)}
            else:
                acc = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
                perf = {"Accuracy": round(acc, 4)}

            st.session_state.model_card = {
                "model": best_model_name,
                "problem": problem_type,
                "mode": training_mode,
                "rows": df.shape[0],
                "features": X.shape[1],
                "target": target_col,
                "performance": perf
            }

            register_model("models/best_model.pkl", best_model_name, 0, X.shape[1], {})

            st.success(f"🏆 Best Model: {best_model_name}")
            st.dataframe(results, use_container_width=True)

    # 🧾 MODEL CARD UI
    if st.session_state.model_card:
        card = st.session_state.model_card
        st.subheader("🧾 Model Card")
        st.json(card)

# ======================================================
# EXPLAINABILITY
# ======================================================
elif page == "🧠 Explainability":
    if st.session_state.model is None:
        st.warning("Train a model first")
        st.stop()

    import shap
    X_sample = st.session_state.X.sample(min(200, len(st.session_state.X)))
    explainer = shap.TreeExplainer(st.session_state.model)
    shap_vals = explainer.shap_values(X_sample)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_vals, X_sample, show=False)
    st.pyplot(fig)






st.subheader("💼 Business Impact Insights")

from src.impact import generate_business_impact

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

insights = generate_business_impact(
    shap_vals,
    X,
    problem_type,
    target_col
)

for i in insights:
    st.info(i)





# ======================================================
# PREDICTION
# ======================================================
elif page == "🔮 Prediction":
    if st.session_state.model is None:
        st.warning("Train a model first")
        st.stop()

    model = st.session_state.model
    schema = st.session_state.feature_schema

    user_date = st.date_input("📅 Select Date")
    inputs = {}

    for col in schema:
        if not col.startswith("Date_"):
            inputs[col] = st.number_input(col, 0.0)

    inputs.update({
        "Date_year": user_date.year,
        "Date_month": user_date.month,
        "Date_day": user_date.day,
        "Date_dayofweek": user_date.weekday(),
        "Date_is_weekend": int(user_date.weekday() >= 5)
    })

    if st.button("Predict"):
        raw = pd.DataFrame([inputs])
        X_pred = prepare_features(raw, profile, training=False, feature_schema=schema)
        pred = model.predict(X_pred)[0]

        if st.session_state.problem_type == "regression":
            std = st.session_state.residual_std or 0
            st.success(f"Prediction: {pred:.2f}")
            st.info(f"Confidence Range: {pred-1.5*std:.2f} – {pred+1.5*std:.2f}")
        else:
            st.success(f"Prediction: {pred}")

# ======================================================
# MODEL REGISTRY
# ======================================================
elif page == "📦 Model Registry":
    st.dataframe(pd.DataFrame(get_all_models()), use_container_width=True)

# ======================================================
# DOWNLOADS
# ======================================================
elif page == "⬇️ Downloads":
    if os.path.exists("models/best_model.pkl"):
        with open("models/best_model.pkl", "rb") as f:
            st.download_button("Download Model", f, "best_model.pkl")


st.subheader("📄 Export Model Report")

if st.button("📥 Download PDF Report"):
    path = generate_pdf_report(
        st.session_state.model_card,
        insights
    )

    with open(path, "rb") as f:
        st.download_button(
            "⬇️ Download Report",
            f,
            "DataPilot_Model_Report.pdf"
        )
