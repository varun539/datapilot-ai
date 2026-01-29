import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from src.pipeline import prepare_features
from src.data_loader import load_csv
from src.eda import (
    basic_profile,
    plot_numeric_distributions,
    plot_correlation_heatmap,
    plot_categorical_counts,
    plot_time_series
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
# CACHE DATA LOADING
# ======================================================
@st.cache_data
def load_cached_csv(file):
    return load_csv(file)

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("🚀 Varun's DataPilot AI")
st.sidebar.caption("End-to-End AutoML Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Data Overview",
        "📈 Visual Analytics",
        "🤖 AutoML",
        "📦 Model Registry",
        "🧠 Explainability",
        "⬇️ Downloads",
        "🔮 Prediction"
    ]
)

st.sidebar.divider()

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
# MAIN HEADER
# ======================================================
st.title("🚀 Varun's DataPilot AI")
st.caption("Built by Varun B — Production AutoML Platform")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    key="main_csv_upload"
)


# ======================================================
# LOAD DATA
# ======================================================
if uploaded_file:
    df = load_cached_csv(uploaded_file)
    profile = basic_profile(df)

    st.sidebar.success("✅ Dataset Loaded")
    st.sidebar.metric("Rows", df.shape[0])
    st.sidebar.metric("Columns", df.shape[1])
else:
    st.info("📂 Upload a CSV file to begin.")
    st.stop()

date_cols = profile.get("datetime_cols", [])

# ======================================================
# 📊 DATA OVERVIEW
# ======================================================
if page == "📊 Data Overview":

    st.header("📊 Dataset Overview")

    score, level, messages = calculate_data_quality(profile)

    c1, c2 = st.columns([1, 2])
    c1.metric("Quality Score", f"{score}/100")
    c2.markdown(f"### {level}")

    if messages:
        for msg in messages:
            st.warning(msg)
    else:
        st.success("✅ Dataset looks healthy!")

    st.dataframe(df.head(), use_container_width=True)

# ======================================================
# 📈 VISUAL ANALYTICS
# ======================================================
elif page == "📈 Visual Analytics":

    st.header("📈 Visual Analytics")

    numeric_cols = profile["numeric_cols"]

    if numeric_cols:
        for fig in plot_numeric_distributions(df, numeric_cols):
            st.pyplot(fig, use_container_width=True)

    heatmap_fig = plot_correlation_heatmap(df, numeric_cols)
    if heatmap_fig:
        st.pyplot(heatmap_fig, use_container_width=True)

    if profile["categorical_cols"]:
        for fig in plot_categorical_counts(df, profile["categorical_cols"]):
            st.pyplot(fig, use_container_width=True)

# ======================================================
# 🤖 AUTOML
# ======================================================
elif page == "🤖 AutoML":

    st.header("🤖 Automated Machine Learning")

    # numeric_cols = profile["numeric_cols"]
    # target_col = st.selectbox("🎯 Select Target Column", numeric_cols)
    # st.session_state.target_col = target_col

    # ✅ Allow numeric + convertible columns
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
        "⚖️ Handle Class Imbalance Automatically (Class Weights)",
        value=True
    )

    # ---------------- TRAIN ----------------
    if st.button("🚀 Train Models"):

        with st.spinner("Training models..."):

            X = prepare_features(
                df_raw=df,
                profile=profile,
                target_col=target_col,
                training=True
            )

            y = df[target_col].fillna(df[target_col].median())

            st.session_state.X = X
            st.session_state.y = y

            # Save schema
            os.makedirs("models", exist_ok=True)
            joblib.dump(X.columns.tolist(), "models/feature_schema.pkl")

            # Detect problem type
            problem_type = detect_problem_type(y)
            st.session_state.problem_type = problem_type

            # ---------------- Imbalance Detection ----------------
            if problem_type == "classification":
                is_imb, ratio = detect_class_imbalance(y)
                if is_imb:
                    st.warning(
                        f"⚠️ Imbalanced dataset detected! "
                        f"Majority class = {ratio*100:.1f}%"
                    )
                else:
                    st.success("✅ Dataset appears balanced.")

            # ---------------- Train Models ----------------
            results_df, best_model_name = train_models(
                X,
                y,
                problem_type,
                handle_imbalance=st.session_state.handle_imbalance
            )

            st.success(f"🏆 Best Model: {best_model_name}")
            st.dataframe(results_df, use_container_width=True)

            model = joblib.load("models/best_model.pkl")

            # ======================================================
            # 📊 CLASSIFICATION DIAGNOSTICS
            # ======================================================
            if problem_type == "classification":

                st.subheader("📊 Classification Diagnostics")

                tscv = TimeSeriesSplit(n_splits=5)

                # ---- Metrics ----
                metrics = {
                    "Accuracy": "accuracy",
                    "F1": "f1_weighted",
                    "Precision": "precision_weighted",
                    "Recall": "recall_weighted"
                }

                cols = st.columns(4)
                for i, (name, metric) in enumerate(metrics.items()):
                    scores = cross_val_score(model, X, y, cv=tscv, scoring=metric)
                    cols[i].metric(name, round(scores.mean(), 4))

                # ---- Confusion Matrix ----
                st.subheader("🧩 Confusion Matrix")
                y_pred = cross_val_predict(model, X, y, cv=tscv)
                cm = confusion_matrix(y, y_pred)

                fig_cm, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(cm)
                disp.plot(ax=ax)
                st.pyplot(fig_cm)

                # ---- ROC Curve (Binary only) ----
                if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
                    st.subheader("📈 ROC Curve")

                    y_prob = cross_val_predict(
                        model,
                        X,
                        y,
                        cv=tscv,
                        method="predict_proba"
                    )[:, 1]

                    fpr, tpr, _ = roc_curve(y, y_prob)
                    roc_auc = auc(fpr, tpr)

                    fig_roc, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                    ax.plot([0, 1], [0, 1], linestyle="--")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    st.pyplot(fig_roc)

            # ---------------- Register Model ----------------
            record = register_model(
                model_path="models/best_model.pkl",
                model_name=best_model_name,
                cv_score=0.0,
                feature_count=X.shape[1],
                best_params={}
            )

            st.info(f"📦 Model registered as version: {record['version_id']}")

    # ---------------- TUNING ----------------
    st.divider()
    st.subheader("⚡ Hyperparameter Tuning")

    if st.button("⚡ Tune Best Model"):

        if st.session_state.X is None:
            st.warning("Train model first.")
            st.stop()

        with st.spinner("Tuning hyperparameters..."):

            base_model = joblib.load("models/best_model.pkl")

            tuned_model, best_params, best_score = tune_best_model(
                base_model,
                st.session_state.X,
                st.session_state.y,
                st.session_state.problem_type
            )

            joblib.dump(tuned_model, "models/best_model.pkl")

            st.success("✅ Model tuned successfully!")
            st.json(best_params)
            st.metric("Best CV Score", best_score)

            record = register_model(
                model_path="models/best_model.pkl",
                model_name=type(tuned_model).__name__,
                cv_score=best_score,
                feature_count=st.session_state.X.shape[1],
                best_params=best_params
            )

            st.info(f"📦 Tuned model registered as version: {record['version_id']}")

# ======================================================
# 🧠 EXPLAINABILITY
# ======================================================


# ================================
# 🧠 EXPLAINABILITY (SAFE)
# ================================
elif page == "🧠 Explainability":

    st.header("🧠 Model Explainability")

    try:
        import shap

        model = joblib.load("models/best_model.pkl")

        # ✅ Take sample
        X_sample_df = st.session_state.X.sample(
            min(200, len(st.session_state.X))
        ).copy()

        # ✅ Force numeric conversion
        X_sample_df = X_sample_df.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(0)

        # ✅ Convert to numpy float32
        X_sample = X_sample_df.values.astype(np.float32)

        feature_names = X_sample_df.columns.tolist()

        st.success("✅ SHAP sample prepared successfully")

        # ✅ Build explainer safely
        explainer = shap.Explainer(model, X_sample, feature_names=feature_names)
        shap_values = explainer(X_sample)

        fig = plt.figure(figsize=(10, 5))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ SHAP failed")
        st.code(str(e))

# ======================================================
# ⬇️ DOWNLOADS
# ======================================================
elif page == "⬇️ Downloads":

    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            st.download_button("⬇️ Download Model", f, "best_model.pkl")

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

        # ---- Single ----
        if mode == "Single Prediction":

            safe_cols = [
                c for c in profile["numeric_cols"]
                if c != st.session_state.target_col
            ]

            user_input = {}
            for col in safe_cols:
                user_input[col] = st.number_input(col, value=0.0)

            if st.button("🎯 Predict"):
                raw_df = pd.DataFrame([user_input])

                X_pred = prepare_features(
                    df_raw=raw_df,
                    profile=profile,
                    training=False,
                    feature_schema=feature_schema
                )

                pred = model.predict(X_pred)[0]
                st.success(f"Prediction: {pred}")

        # ---- Batch ----
        else:
        

            batch_file = st.file_uploader(
            "Upload CSV for Batch Prediction",
            type=["csv"],
             key="batch_csv_upload"
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

                st.dataframe(batch_df.head())
                csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Predictions", csv, "predictions.csv")

    except Exception as e:
        st.warning("Train model first.")
        st.code(str(e))

# ======================================================
# 📦 MODEL REGISTRY
# ======================================================
elif page == "📦 Model Registry":

    st.header("📦 Model Registry")

    registry = get_all_models()
    df_registry = pd.DataFrame(registry).sort_values(
        by="created_at",
        ascending=False
    )

    st.dataframe(df_registry, use_container_width=True)

    selected_version = st.selectbox(
        "Select version",
        df_registry["version_id"].tolist()
    )

    selected_row = df_registry[
        df_registry["version_id"] == selected_version
    ].iloc[0]

    if st.button("🚀 Load This Model"):
        joblib.dump(
            joblib.load(selected_row["model_path"]),
            "models/best_model.pkl"
        )
        st.success(f"Loaded model version {selected_version}")
