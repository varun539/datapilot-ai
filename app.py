import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

from src.pipeline import prepare_features
from src.data_loader import load_csv
from src.eda import basic_profile, plot_numeric_distributions, plot_correlation_heatmap
from src.automl import (
    detect_problem_type,
    train_models,
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

@st.cache_data
def load_cached_csv(file):
    return load_csv(file)

# ======================================================
# SESSION STATE
# ======================================================
for k in [
    "X", "y", "model", "problem_type", "target_col",
    "training_mode", "feature_schema",
    "model_card", "business_insights",
    "residual_std"
]:
    st.session_state.setdefault(k, None)

st.session_state.setdefault("handle_imbalance", True)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🚀 Varun's DataPilot AI")
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
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.stop()

df = load_cached_csv(uploaded_file)
profile = basic_profile(df)

# ======================================================
# DATA OVERVIEW
# ======================================================
if page == "📊 Data Overview":
    score, level, msgs = calculate_data_quality(profile)
    st.metric("Quality Score", score)
    for m in msgs:
        st.warning(m)
    st.dataframe(df.head())

# ======================================================
# VISUALS
# ======================================================
elif page == "📈 Visual Analytics":
    for fig in plot_numeric_distributions(df, profile["numeric_cols"]):
        st.pyplot(fig)

# ======================================================
# AUTOML
# ======================================================
elif page == "🤖 AutoML":

    numeric_targets = []
    for c in df.columns:
        try:
            pd.to_numeric(df[c])
            numeric_targets.append(c)
        except:
            pass

    target_col = st.selectbox("🎯 Select Target Column", numeric_targets)

    if st.button("🚀 Train Models"):
        with st.spinner("Training..."):

            X = prepare_features(df, profile, target_col, training=True)
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(df[target_col].median())

            leak, feats = detect_data_leakage(X, y)
            if leak:
                st.error("Data leakage detected")
                st.stop()

            problem_type = detect_problem_type(y)
            training_mode = detect_training_mode(df, target_col, profile)

            results, best_model_name = train_models(
                X, y, problem_type, st.session_state.handle_imbalance
            )

            model = joblib.load("models/best_model.pkl")

            # ===============================
            # CROSS-VALIDATION (🔥 CORRECT)
            # ===============================
            if problem_type == "regression":
                cv = KFold(5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                cv_name = "R2 (CV)"
            else:
                cv = StratifiedKFold(5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                cv_name = "Accuracy (CV)"

            cv_mean = scores.mean()
            cv_std = scores.std()

            # ===============================
            # SAVE STATE
            # ===============================
            st.session_state.update({
                "X": X,
                "y": y,
                "model": model,
                "problem_type": problem_type,
                "training_mode": training_mode,
                "feature_schema": X.columns.tolist(),
            })

            if problem_type == "regression":
                preds = model.predict(X)
                st.session_state.residual_std = np.std(y - preds)

            # ===============================
            # MODEL CARD (FINAL)
            # ===============================
            st.session_state.model_card = {
                "model": best_model_name,
                "problem": problem_type,
                "mode": training_mode,
                "rows": df.shape[0],
                "features": X.shape[1],
                "target": target_col,
                "performance": {
                    cv_name: round(cv_mean, 4),
                    "CV Std": round(cv_std, 4)
                }
            }

            register_model(
                "models/best_model.pkl",
                best_model_name,
                round(cv_mean, 4),
                X.shape[1],
                {}
            )

            log_experiment(st.session_state.model_card)

            # ===============================
            # BUSINESS IMPACT
            # ===============================
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            insights = generate_business_impact(shap_vals, X, problem_type, target_col)
            st.session_state.business_insights = insights

            # ===============================
            # UI OUTPUT
            # ===============================
            st.success(f"🏆 Best Model: {best_model_name}")
            st.metric(cv_name, f"{cv_mean:.4f}", delta=f"±{cv_std:.4f}")
            st.dataframe(results)

            st.subheader("💼 Business Impact")
            for i in insights:
                st.info(i)

# ======================================================
# PREDICTION (SINGLE + BATCH)
# ======================================================
elif page == "🔮 Prediction":
    if st.session_state.model is None:
        st.stop()

    model = st.session_state.model
    schema = st.session_state.feature_schema

    mode = st.radio("Mode", ["Single", "Batch"])

    if mode == "Single":
        date = st.date_input("Date")
        inputs = {c: st.number_input(c, 0.0) for c in schema if not c.startswith("Date_")}
        inputs.update({
            "Date_year": date.year,
            "Date_month": date.month,
            "Date_day": date.day,
            "Date_dayofweek": date.weekday(),
            "Date_is_weekend": int(date.weekday() >= 5)
        })

        if st.button("Predict"):
            Xp = prepare_features(pd.DataFrame([inputs]), profile, False, schema)
            pred = model.predict(Xp)[0]
            st.success(pred)

    else:
        f = st.file_uploader("Upload CSV")
        if f:
            bdf = load_cached_csv(f)
            Xb = prepare_features(bdf, profile, False, schema)
            preds = model.predict(Xb)
            bdf["prediction"] = preds
            st.dataframe(bdf.head())
            st.download_button("Download", bdf.to_csv(index=False), "preds.csv")

# ======================================================
# EXPERIMENTS
# ======================================================
elif page == "🧪 Experiment History":
    exp = load_experiments()
    st.dataframe(pd.DataFrame(exp))

# ======================================================
# REGISTRY
# ======================================================
elif page == "📦 Model Registry":
    st.dataframe(pd.DataFrame(get_all_models()))

# ======================================================
# DOWNLOADS
# ======================================================
elif page == "⬇️ Downloads":
    if st.session_state.model_card:
        path = generate_pdf_report(
            st.session_state.model_card,
            st.session_state.business_insights
        )
        with open(path, "rb") as f:
            st.download_button("Download Report", f, "report.pdf")
