import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import shap

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
from src.impact import generate_business_impact
from src.report import generate_pdf_report
from src.experiments import log_experiment, load_experiments


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
# SESSION STATE INIT
# ======================================================
STATE_KEYS = [
    "X", "y", "model", "problem_type", "target_col",
    "training_mode", "feature_schema",
    "model_card", "business_insights",
    "residual_std"
]

for k in STATE_KEYS:
    st.session_state.setdefault(k, None)

st.session_state.setdefault("handle_imbalance", True)

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
        "🧪 Experiment History",
        "📦 Model Registry",
        "⬇️ Downloads"
    ]
)

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
# 📊 DATA OVERVIEW
# ======================================================
if page == "📊 Data Overview":
    score, level, messages = calculate_data_quality(profile)
    st.metric("Quality Score", f"{score}/100")
    st.markdown(f"### {level}")
    for m in messages:
        st.warning(m)
    st.dataframe(df.head(), use_container_width=True)

# ======================================================
# 📈 VISUAL ANALYTICS
# ======================================================
elif page == "📈 Visual Analytics":
    for fig in plot_numeric_distributions(df, profile["numeric_cols"]):
        st.pyplot(fig, use_container_width=True)

    heatmap = plot_correlation_heatmap(df, profile["numeric_cols"])
    if heatmap:
        st.pyplot(heatmap, use_container_width=True)

# ======================================================
# 🤖 AUTOML
# ======================================================
elif page == "🤖 AutoML":
    st.header("🤖 Automated Machine Learning")

    candidate_targets = []
    for c in df.columns:
        try:
            pd.to_numeric(df[c])
            candidate_targets.append(c)
        except:
            pass

    target_col = st.selectbox("🎯 Select Target Column", candidate_targets)
    st.session_state.target_col = target_col

    st.session_state.handle_imbalance = st.checkbox(
        "Handle Class Imbalance Automatically", True
    )

    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):

            X = prepare_features(df, profile, target_col, training=True)
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(df[target_col].median())

            # 🚨 Data Leakage Guard
            leak, leaked = detect_data_leakage(X, y)
            if leak:
                st.error("⚠️ Data leakage detected. Training stopped.")
                for f, c in leaked:
                    st.write(f"{f} → corr={c:.3f}")
                st.stop()

            problem_type = detect_problem_type(y)
            training_mode = detect_training_mode(df, target_col, profile)

            results, best_model_name = train_models(
                X, y, problem_type, st.session_state.handle_imbalance
            )

            model = joblib.load("models/best_model.pkl")

            # Store session
            st.session_state.update({
                "X": X,
                "y": y,
                "model": model,
                "problem_type": problem_type,
                "training_mode": training_mode,
                "feature_schema": X.columns.tolist()
            })

            # Confidence stats
            if problem_type == "regression":
                preds = model.predict(X)
                st.session_state.residual_std = np.std(y - preds)

            # Model card
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

            # Register + log experiment
            register_model("models/best_model.pkl", best_model_name, 0, X.shape[1], {})
            log_experiment({
                "model": best_model_name,
                "problem": problem_type,
                "mode": training_mode,
                "rows": df.shape[0],
                "features": X.shape[1],
                "target": target_col,
                "metrics": perf
            })

            # Business impact
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            insights = generate_business_impact(shap_vals, X, problem_type, target_col)
            st.session_state.business_insights = insights

            st.success(f"🏆 Best Model: {best_model_name}")
            st.dataframe(results, use_container_width=True)

            st.subheader("💼 Business Impact")
            for i in insights:
                st.info(i)

# ======================================================
# 🧠 EXPLAINABILITY
# ======================================================
elif page == "🧠 Explainability":
    if st.session_state.model is None:
        st.warning("Train a model first")
        st.stop()

    Xs = st.session_state.X.sample(min(200, len(st.session_state.X)))
    explainer = shap.TreeExplainer(st.session_state.model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]

    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(sv, Xs, show=False)
    st.pyplot(fig)

# ======================================================
# 🔮 PREDICTION
# ======================================================
# elif page == "🔮 Prediction":
#     if st.session_state.model is None:
#         st.warning("Train a model first")
#         st.stop()

#     model = st.session_state.model
#     schema = st.session_state.feature_schema

#     date = st.date_input("📅 Select Date")
#     inputs = {c: st.number_input(c, 0.0) for c in schema if not c.startswith("Date_")}
#     inputs.update({
#         "Date_year": date.year,
#         "Date_month": date.month,
#         "Date_day": date.day,
#         "Date_dayofweek": date.weekday(),
#         "Date_is_weekend": int(date.weekday() >= 5)
#     })

#     if st.button("Predict"):
#         Xp = prepare_features(pd.DataFrame([inputs]), profile, training=False, feature_schema=schema)
#         pred = model.predict(Xp)[0]

#         if st.session_state.problem_type == "regression":
#             std = st.session_state.residual_std or 0
#             st.success(f"Prediction: {pred:.2f}")
#             st.info(f"Confidence Range: {pred-1.5*std:.2f} – {pred+1.5*std:.2f}")
#         else:
#             prob = model.predict_proba(Xp)[0]
#             st.success(f"Class: {np.argmax(prob)}")
#             st.info(f"Confidence: {np.max(prob)*100:.1f}%")




elif page == "🔮 Prediction":
    if st.session_state.model is None:
        st.warning("Train a model first")
        st.stop()

    model = st.session_state.model
    schema = st.session_state.feature_schema
    problem_type = st.session_state.problem_type

    mode = st.radio(
        "Prediction Mode",
        ["Single Prediction", "Batch CSV Prediction"]
    )

    # ======================================================
    # 🧍 SINGLE PREDICTION (already good)
    # ======================================================
    if mode == "Single Prediction":

        date = st.date_input("📅 Select Date")

        inputs = {
            c: st.number_input(c, 0.0)
            for c in schema
            if not c.startswith("Date_")
        }

        inputs.update({
            "Date_year": date.year,
            "Date_month": date.month,
            "Date_day": date.day,
            "Date_dayofweek": date.weekday(),
            "Date_is_weekend": int(date.weekday() >= 5)
        })

        if st.button("🎯 Predict"):
            Xp = prepare_features(
                pd.DataFrame([inputs]),
                profile,
                training=False,
                feature_schema=schema
            )

            pred = model.predict(Xp)[0]

            if problem_type == "regression":
                std = st.session_state.residual_std or 0
                st.success(f"Prediction: {pred:.2f}")
                st.info(f"Confidence Range: {pred-1.5*std:.2f} – {pred+1.5*std:.2f}")
            else:
                prob = model.predict_proba(Xp)[0]
                st.success(f"Class: {np.argmax(prob)}")
                st.info(f"Confidence: {np.max(prob)*100:.1f}%")

    # ======================================================
    # 📦 BATCH CSV PREDICTION (NEW 🔥)
    # ======================================================
    else:
        st.subheader("📦 Batch CSV Prediction")

        batch_file = st.file_uploader(
            "Upload CSV for Batch Prediction",
            type=["csv"],
            key="batch_upload"
        )

        if batch_file:
            batch_df = load_cached_csv(batch_file)

            st.info(f"Rows uploaded: {batch_df.shape[0]}")

            # 🔧 Run same feature pipeline
            X_batch = prepare_features(
                batch_df,
                profile,
                training=False,
                feature_schema=schema
            )

            preds = model.predict(X_batch)
            result_df = batch_df.copy()
            result_df["prediction"] = preds

            # =============================
            # Confidence (optional but 🔥)
            # =============================
            if problem_type == "regression":
                std = st.session_state.residual_std or 0
                result_df["lower_bound"] = preds - 1.5 * std
                result_df["upper_bound"] = preds + 1.5 * std

            elif problem_type == "classification":
                probs = model.predict_proba(X_batch)
                result_df["confidence"] = probs.max(axis=1)

            st.success("✅ Batch prediction completed")
            st.dataframe(result_df.head(20), use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                csv,
                "batch_predictions.csv",
                mime="text/csv"
            )

# ======================================================
# 🧪 EXPERIMENT HISTORY
# ======================================================
elif page == "🧪 Experiment History":
    st.header("🧪 Training Experiments")
    exp = load_experiments()
    if not exp:
        st.info("No experiments logged yet.")
    else:
        st.dataframe(pd.DataFrame(exp), use_container_width=True)

# ======================================================
# 📦 MODEL REGISTRY
# ======================================================
elif page == "📦 Model Registry":
    st.dataframe(pd.DataFrame(get_all_models()), use_container_width=True)

# ======================================================
# ⬇️ DOWNLOADS
# ======================================================
elif page == "⬇️ Downloads":
    if os.path.exists("models/best_model.pkl"):
        with open("models/best_model.pkl", "rb") as f:
            st.download_button("Download Model", f, "best_model.pkl")

    if st.session_state.model_card and st.session_state.business_insights:
        path = generate_pdf_report(
            st.session_state.model_card,
            st.session_state.business_insights
        )
        with open(path, "rb") as f:
            st.download_button("Download PDF Report", f, "DataPilot_Report.pdf")
