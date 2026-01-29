import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from src.pipeline import prepare_features
from src.data_loader import load_csv
from src.eda import (
    basic_profile,
    plot_numeric_distributions,
    plot_correlation_heatmap,
    plot_categorical_counts
)
from src.automl import (
    detect_problem_type,
    train_models,
    tune_best_model,
    detect_class_imbalance
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
st.sidebar.caption("End-to-End AutoML Platform")

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
# SESSION STATE
# ======================================================
if "X" not in st.session_state:
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.problem_type = None
    st.session_state.target_col = None
    st.session_state.handle_imbalance = True

# ======================================================
# HEADER
# ======================================================
st.title("🚀 Varun's DataPilot AI")
st.caption("Production AutoML Platform")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    key="main_upload"
)

# ======================================================
# LOAD DATA
# ======================================================
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
    st.header("Dataset Overview")

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
    st.header("Visual Analytics")

    for fig in plot_numeric_distributions(df, profile["numeric_cols"]):
        st.pyplot(fig, use_container_width=True)

    heatmap = plot_correlation_heatmap(df, profile["numeric_cols"])
    if heatmap:
        st.pyplot(heatmap, use_container_width=True)

# ======================================================
# AUTOML
# ======================================================
elif page == "🤖 AutoML":
    st.header("Automated Machine Learning")

    # ✅ NUMERIC-CONVERTIBLE TARGETS
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
        "Handle Class Imbalance Automatically",
        value=True
    )

    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):

            X = prepare_features(
                df_raw=df,
                profile=profile,
                target_col=target_col,
                training=True
            )

            y = pd.to_numeric(df[target_col], errors="coerce").fillna(df[target_col].median())

            st.session_state.X = X
            st.session_state.y = y

            os.makedirs("models", exist_ok=True)
            joblib.dump(X.columns.tolist(), "models/feature_schema.pkl")

            problem_type = detect_problem_type(y)
            st.session_state.problem_type = problem_type

            if problem_type == "classification":
                is_imb, ratio = detect_class_imbalance(y)
                if is_imb:
                    st.warning(f"Imbalanced data detected ({ratio*100:.1f}% majority)")

            results_df, best_model = train_models(
                X,
                y,
                problem_type,
                handle_imbalance=st.session_state.handle_imbalance
            )

            st.success(f"Best Model: {best_model}")
            st.dataframe(results_df, use_container_width=True)

            register_model(
                model_path="models/best_model.pkl",
                model_name=best_model,
                cv_score=0.0,
                feature_count=X.shape[1],
                best_params={}
            )

# ======================================================
# EXPLAINABILITY
# ======================================================
elif page == "🧠 Explainability":
    st.header("Model Explainability")

    if st.session_state.X is None:
        st.warning("Train a model first")
        st.stop()

    import shap

    model = joblib.load("models/best_model.pkl")

    X_sample = (
        st.session_state.X
        .sample(min(200, len(st.session_state.X)))
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values.astype(np.float32)
    )

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)

# ======================================================
# PREDICTION
# ======================================================
elif page == "🔮 Prediction":
    st.header("Prediction")

    if not os.path.exists("models/best_model.pkl"):
        st.warning("Train a model first")
        st.stop()

    model = joblib.load("models/best_model.pkl")
    schema = joblib.load("models/feature_schema.pkl")

    mode = st.radio("Mode", ["Single", "Batch"])

    if mode == "Single":
        inputs = {c: st.number_input(c, 0.0) for c in schema}
        if st.button("Predict"):
            df_in = pd.DataFrame([inputs])
            X_pred = prepare_features(
                df_raw=df_in,
                profile=profile,
                training=False,
                feature_schema=schema
            )
            st.success(f"Prediction: {model.predict(X_pred)[0]}")

    else:
        batch = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")
        if batch:
            batch_df = load_cached_csv(batch)
            X_batch = prepare_features(
                df_raw=batch_df,
                profile=profile,
                training=False,
                feature_schema=schema
            )
            batch_df["prediction"] = model.predict(X_batch)
            st.dataframe(batch_df.head())
            st.download_button(
                "Download Predictions",
                batch_df.to_csv(index=False).encode(),
                "predictions.csv"
            )

# ======================================================
# MODEL REGISTRY
# ======================================================
elif page == "📦 Model Registry":
    st.header("Model Registry")
    df_reg = pd.DataFrame(get_all_models())
    st.dataframe(df_reg, use_container_width=True)

# ======================================================
# DOWNLOADS
# ======================================================
elif page == "⬇️ Downloads":
    if os.path.exists("models/best_model.pkl"):
        with open("models/best_model.pkl", "rb") as f:
            st.download_button("Download Model", f, "best_model.pkl")
 
