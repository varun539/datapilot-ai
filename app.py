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
# elif page == "🧠 Explainability":
#     st.header("Model Explainability")

#     if st.session_state.X is None:
#         st.warning("Train a model first")
#         st.stop()

#     import shap

#     model = joblib.load("models/best_model.pkl")

#     X_sample = (
#         st.session_state.X
#         .sample(min(200, len(st.session_state.X)))
#         .apply(pd.to_numeric, errors="coerce")
#         .fillna(0)
#         .values.astype(np.float32)
#     )

#     explainer = shap.Explainer(model, X_sample)
#     shap_values = explainer(X_sample)

#     fig = plt.figure(figsize=(10, 5))
#     shap.summary_plot(shap_values, X_sample, show=False)
#     st.pyplot(fig)



# ======================================================
# 🧠 EXPLAINABILITY (SHAP — FIXED)
# # ======================================================
# elif page == "🧠 Explainability":

#     st.header("🧠 Model Explainability (SHAP)")

#     try:
#         import shap

#         model = joblib.load("models/best_model.pkl")

#         # 👉 Sample data
#         X_sample_df = st.session_state.X.sample(
#             min(200, len(st.session_state.X)),
#             random_state=42
#         ).copy()

#         # 👉 Force numeric safety
#         X_sample_df = X_sample_df.apply(
#             pd.to_numeric,
#             errors="coerce"
#         ).fillna(0)

#         st.success("✅ SHAP sample prepared")

#         # 👉 KEEP AS DATAFRAME (VERY IMPORTANT)
#         explainer = shap.Explainer(model, X_sample_df)
#         shap_values = explainer(X_sample_df)

#         fig = plt.figure(figsize=(10, 5))
#         shap.summary_plot(
#             shap_values,
#             X_sample_df,
#             show=False
#         )

#         st.pyplot(fig, use_container_width=True)

#     except Exception as e:
#         st.error("❌ SHAP failed")
#         st.code(str(e))


elif page == "🧠 Explainability":

    st.header("🧠 Model Explainability (SHAP)")

    try:
        import shap

        model = joblib.load("models/best_model.pkl")

        # Take safe sample
        X_sample_df = st.session_state.X.sample(
            min(200, len(st.session_state.X)),
            random_state=42
        ).copy()

        # Force numeric safety
        X_sample_df = X_sample_df.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)

        st.success("✅ SHAP sample prepared")

        # 🔥 IMPORTANT FIX — use TreeExplainer explicitly
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_df)

        # =============================
        # Regression OR Binary Class
        # =============================
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]  # positive class
        else:
            shap_values_to_plot = shap_values

        fig = plt.figure(figsize=(10, 5))
        shap.summary_plot(
            shap_values_to_plot,
            X_sample_df,
            show=False
        )

        st.pyplot(fig)

    except Exception as e:
        st.error("❌ SHAP failed")
        st.code(str(e))







# ======================================================
# 🔮 PREDICTION
# ======================================================
elif page == "🔮 Prediction":

    st.header("🔮 Prediction")

    try:
        model = joblib.load("models/best_model.pkl")
        feature_schema = joblib.load("models/feature_schema.pkl")

        mode = st.radio(
            "Prediction Mode",
            ["Single Prediction", "Batch CSV Prediction"]
        )

        # =====================================
        # 🧍 SINGLE PREDICTION
        # =====================================
        if mode == "Single Prediction":

            st.subheader("🧍 Single Prediction")

            # 👉 ONE date picker for humans
            user_date = st.date_input("📅 Select Date")

            # 👉 Exclude engineered date features from UI
            safe_cols = [
                c for c in feature_schema
                if not c.startswith("Date_")
            ]

            user_input = {}

            for col in safe_cols:
                user_input[col] = st.number_input(col, value=0.0)

            # 👉 Backend date engineering (MODEL SEES THIS)
            date_features = {
                "Date_year": user_date.year,
                "Date_month": user_date.month,
                "Date_day": user_date.day,
                "Date_dayofweek": user_date.weekday(),
                "Date_is_weekend": int(user_date.weekday() >= 5)
            }

            user_input.update(date_features)

            if st.button("🎯 Predict"):

                raw_df = pd.DataFrame([user_input])

                X_pred = prepare_features(
                    df_raw=raw_df,
                    profile=profile,
                    training=False,
                    feature_schema=feature_schema
                )

                prediction = model.predict(X_pred)[0]
                st.success(f"📈 Prediction Result: {prediction}")

        # =====================================
        # 📦 BATCH PREDICTION
        # =====================================
        else:

            st.subheader("📦 Batch Prediction")

            batch_file = st.file_uploader(
                "Upload CSV for Batch Prediction",
                type=["csv"],
                key="batch_prediction_upload"
            )

            if batch_file:
                batch_df = load_cached_csv(batch_file)

                X_batch = prepare_features(
                    df_raw=batch_df,
                    profile=profile,
                    training=False,
                    feature_schema=feature_schema
                )

                preds = model.predict(X_batch)
                batch_df["prediction"] = preds

                st.dataframe(batch_df.head(20), use_container_width=True)

                csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions",
                    csv,
                    "predictions.csv"
                )

    except Exception as e:
        st.warning("⚠️ Train a model first.")
        st.code(str(e))









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
 
