
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# Load OpenAI key from Streamlit secrets
api_key = st.secrets.get("OPENAI_API_KEY", None)

from src.pipeline import prepare_features
from src.data_loader import load_csv
from src.eda import (
    basic_profile,
    plot_numeric_distributions,
    plot_correlation_heatmap
)
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
from src.agent import (
    generate_agent_narrative,
    chat_with_data,
    suggest_target_column,
    diagnose_dataset
)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="DataPilot AI",
    layout="wide",
    page_icon="🚀"
)

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
    "X", "y", "model", "problem_type", "target_col",
    "training_mode", "feature_schema",
    "model_card", "business_insights", "residual_std",
    "agent_narrative", "chat_history", "shap_top_features"
]

for k in STATE_KEYS:
    st.session_state.setdefault(k, None)

st.session_state.setdefault("chat_history", [])

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🚀 DataPilot AI")
st.sidebar.caption("Agentic AutoML Platform by Varun B")

with st.sidebar.expander("🔑 AI Status", expanded=False):

    if api_key:
        st.success("OpenAI API Connected")
    else:
        st.warning("OpenAI API key not configured")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Data Overview",
        "📈 Visual Analytics",
        "🤖 AutoML",
        "🧠 Explainability",
        "🤖 AI Insights",
        "💬 Chat with Data",
        "🔮 Prediction",
        "🧪 Experiment History",
        "📦 Model Registry",
        "⬇️ Downloads"
    ]
)

# ======================================================
# HEADER
# ======================================================

st.title("🚀 DataPilot AI")

st.markdown(
    "Upload your dataset → Train models automatically → Get AI-powered insights."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin.")
    st.stop()

df = load_cached_csv(uploaded_file)

profile = basic_profile(df)

st.sidebar.success("Dataset Loaded")
st.sidebar.metric("Rows", df.shape[0])
st.sidebar.metric("Columns", df.shape[1])




# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="DataPilot AI", layout="wide", page_icon="🚀")

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
    "X", "y", "model", "problem_type", "target_col",
    "training_mode", "feature_schema",
    "model_card", "business_insights", "residual_std",
    "agent_narrative", "chat_history", "shap_top_features"
]
for k in STATE_KEYS:
    st.session_state.setdefault(k, None)

st.session_state.setdefault("handle_imbalance", True)
st.session_state.setdefault("chat_history", [])

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🚀 DataPilot AI")
st.sidebar.caption("Agentic AutoML Platform by Varun B")
if not uploaded_file:
    st.info("Upload a CSV to begin.")
    st.stop()

df = load_cached_csv(uploaded_file)
profile = basic_profile(df)

st.sidebar.success("✅ Dataset Loaded")
st.sidebar.metric("Rows", df.shape[0])
st.sidebar.metric("Columns", df.shape[1])

# ======================================================
# DATA OVERVIEW
# ======================================================

if page == "📊 Data Overview":

    score, level, messages = calculate_data_quality(profile)

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric("Quality Score", f"{score}/100")
        st.markdown(f"### {level}")

        for m in messages:
            st.warning(m)

    with col2:

        if api_key:

            if st.button("AI Dataset Diagnosis"):

                with st.spinner("Analyzing dataset..."):

                    diagnosis = diagnose_dataset(
                        api_key,
                        profile,
                        score,
                        messages
                    )

                st.info(diagnosis)

        else:

            st.info("AI features unavailable")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)

    # ======================================================
    # QUICK DATA INSIGHTS
    # ======================================================

    st.divider()

    st.subheader("⚡ Quick Data Insights")

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr().abs()

        corr.values[[np.arange(corr.shape[0])]*2] = 0

        top_corr = corr.unstack().sort_values(
            ascending=False
        ).drop_duplicates()

        if not top_corr.empty:

            f1, f2 = top_corr.index[0]
            value = top_corr.iloc[0]

            st.info(
                f"Strongest relationship detected: "
                f"**{f1} ↔ {f2}** (corr = {value:.2f})"
            )

    missing = df.isnull().sum().sum()

    if missing > 0:

        st.warning(f"{missing} missing values detected")

    else:

        st.success("No missing values detected")

    st.success(
        f"Dataset contains {df.shape[0]} rows "
        f"and {df.shape[1]} columns"
    )

    # ======================================================
    # TARGET SUGGESTION
    # ======================================================

    if api_key:

        st.divider()

        st.subheader("🎯 AI Target Column Suggestion")

        if st.button("Suggest Target Column"):

            with st.spinner("Analyzing dataset..."):

                suggested = suggest_target_column(
                    api_key,
                    df.columns.tolist(),
                    df
                )

            st.success(f"Suggested target: **{suggested}**")












# ======================================================
# 📈 VISUAL ANALYTICS
# ======================================================
elif page == "📈 Visual Analytics":
    st.header("📈 Visual Analytics")

    for fig in plot_numeric_distributions(df, profile["numeric_cols"]):
        st.pyplot(fig, use_container_width=True)

    numeric_for_corr = [
        c for c in profile["numeric_cols"]
        if c in df.columns and df[c].nunique() > 1
    ]

    if len(numeric_for_corr) >= 2:
        fig = plot_correlation_heatmap(df, numeric_for_corr)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Not enough numeric features for correlation heatmap.")

# ======================================================
# 🤖 AUTOML
# ======================================================
elif page == "🤖 AutoML":
    st.header("🤖 Automated Machine Learning")

    # Smart target column filter — removes ID/useless columns
    useless_keywords = [
        "id", "uuid", "index", "code", "number", "row",
        "order", "invoice", "record", "key", "ref", "postal",
        "zip", "phone", "email"
    ]

    numeric_targets = []
    for c in df.columns:
        col_lower = c.lower().replace(" ", "_")
        # Skip if name looks like ID/useless
        if any(k in col_lower for k in useless_keywords):
            continue
        # Skip high cardinality numeric (likely ID numbers)
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique() / len(df) > 0.8:
                continue
            numeric_targets.append(c)
        else:
            try:
                pd.to_numeric(df[c])
                numeric_targets.append(c)
            except:
                pass

    # Fallback — if nothing left show all numeric
    if not numeric_targets:
        numeric_targets = df.select_dtypes(include="number").columns.tolist()

    target_col = st.selectbox("🎯 Select Target Column", numeric_targets)
    st.session_state.target_col = target_col

    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):

            X = prepare_features(df, profile, target_col, training=True)
            y = pd.to_numeric(df[target_col], errors="coerce").fillna(df[target_col].median())

            leak, leaked = detect_data_leakage(X, y)
            if leak:
                st.error("⚠️ Data leakage detected. Training stopped.")
                for f, c in leaked:
                    st.write(f"{f} → corr={c}")
                st.stop()

            problem_type = detect_problem_type(y)
            training_mode = detect_training_mode(df, target_col, profile)

            results, best_model_name = train_models(X, y, problem_type)
            model = joblib.load("models/best_model.pkl")

            st.session_state.update({
                "X": X,
                "y": y,
                "model": model,
                "problem_type": problem_type,
                "training_mode": training_mode,
                "feature_schema": X.columns.tolist()
            })

            # Cross Validation
            if problem_type == "regression":
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                cv_metric = "R2 (CV)"
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                cv_metric = "Accuracy (CV)"

            cv_mean, cv_std = scores.mean(), scores.std()

            if problem_type == "regression":
                preds = model.predict(X)
                st.session_state.residual_std = np.std(y - preds)

            # SHAP
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            # Store top SHAP features for agent
            mean_abs = np.abs(shap_vals).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:5]
            shap_top = [(X.columns[i], round(float(mean_abs[i]), 4)) for i in top_idx]
            st.session_state.shap_top_features = shap_top

            insights = generate_business_impact(shap_vals, X, problem_type, target_col)
            st.session_state.business_insights = insights

            model_card = {
                "model": best_model_name,
                "problem": problem_type,
                "mode": training_mode,
                "rows": df.shape[0],
                "features": X.shape[1],
                "target": target_col,
                "performance": {
                    cv_metric: round(cv_mean, 4),
                    "CV Std": round(cv_std, 4)
                }
            }
            st.session_state.model_card = model_card

            register_model("models/best_model.pkl", best_model_name, cv_mean, X.shape[1], {})
            log_experiment(model_card)

            st.success(f"🏆 Best Model: {best_model_name}")
            st.metric(cv_metric, f"{cv_mean:.4f}", delta=f"±{cv_std:.4f}")
            st.dataframe(results, use_container_width=True)

            st.subheader("💼 Business Impact")
            for i in insights:
                st.info(i)

            # Auto-generate AI narrative if key present
            if api_key:
                st.divider()
                st.subheader("🤖 AI Narrative (Auto-Generated)")
                with st.spinner("Generating executive insights..."):
                    narrative = generate_agent_narrative(
                        api_key, model_card, insights,
                        profile, shap_top, problem_type, target_col
                    )
                    st.session_state.agent_narrative = narrative
                st.markdown(narrative)
                st.caption("Go to 💬 Chat with Data to ask follow-up questions")

# ======================================================
# 🧠 EXPLAINABILITY
# ======================================================
elif page == "🧠 Explainability":
    if st.session_state.model is None:
        st.warning("Train a model first")
        st.stop()

    Xs = (
        st.session_state.X
        .sample(min(200, len(st.session_state.X)), random_state=42)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    explainer = shap.TreeExplainer(st.session_state.model)
    shap_vals = explainer.shap_values(Xs)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_vals, Xs, show=False)
    st.pyplot(fig)

# ======================================================
# 🤖 AI INSIGHTS — NEW PAGE
# ======================================================
elif page == "🤖 AI Insights":
    st.header("🤖 AI-Powered Insights")

    if st.session_state.model_card is None:
        st.warning("Train a model first to generate AI insights")
        st.stop()

    if not api_key:
        st.warning("Add your OpenAI API key in the sidebar to use this feature")
        st.stop()

    # Show existing narrative or generate new one
    if st.session_state.agent_narrative:
        st.markdown("### 📋 Executive Analysis")
        st.markdown(st.session_state.agent_narrative)
        st.divider()
        if st.button("🔄 Regenerate Analysis"):
            st.session_state.agent_narrative = None
            st.rerun()
    else:
        if st.button("✨ Generate AI Analysis"):
            with st.spinner("GPT-4o is analyzing your model results..."):
                narrative = generate_agent_narrative(
                    api_key,
                    st.session_state.model_card,
                    st.session_state.business_insights or [],
                    profile,
                    st.session_state.shap_top_features or [],
                    st.session_state.problem_type,
                    st.session_state.target_col
                )
                st.session_state.agent_narrative = narrative
            st.markdown(narrative)

    # Show raw insights for comparison
    if st.session_state.business_insights:
        with st.expander("View Basic Insights (pre-AI)"):
            for i in st.session_state.business_insights:
                st.info(i)

# ======================================================
# 💬 CHAT WITH DATA — NEW PAGE
# ======================================================
elif page == "💬 Chat with Data":
    st.header("💬 Chat with Your Data")
    st.caption("Ask anything about your dataset, model, or predictions")

    if st.session_state.model_card is None:
        st.warning("Train a model first to enable chat")
        st.stop()

    if not api_key:
        st.warning("Add your OpenAI API key in the sidebar to use chat")
        st.stop()

    # Fix: ensure chat_history is always a list
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

    # Suggested questions
    if not st.session_state.chat_history:
        st.subheader("💡 Try asking:")
        suggestions = [
            f"What drives {st.session_state.target_col} the most?",
            "Is this model good enough for production?",
            "What should the business do based on these results?",
            "Which features should we focus on improving?",
            "What are the biggest risks with this model?"
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"suggestion_{i}"):
                # Treat as user input
                with st.spinner("Thinking..."):
                    response = chat_with_data(
                        api_key, s,
                        st.session_state.chat_history,
                        st.session_state.model_card,
                        profile, df,
                        st.session_state.problem_type,
                        st.session_state.target_col,
                        st.session_state.business_insights or []
                    )
                st.session_state.chat_history.append({"role": "user", "content": s})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

    # Chat input
    user_input = st.chat_input("Ask about your data or model...")
    if user_input:
        st.chat_message("user").write(user_input)
        with st.spinner("Thinking..."):
            response = chat_with_data(
                api_key, user_input,
                st.session_state.chat_history,
                st.session_state.model_card,
                profile, df,
                st.session_state.problem_type,
                st.session_state.target_col,
                st.session_state.business_insights or []
            )
        st.chat_message("assistant").write(response)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ======================================================
# 🔮 PREDICTION
# ======================================================
elif page == "🔮 Prediction":
    if st.session_state.model is None:
        st.warning("Train a model first")
        st.stop()

    model = st.session_state.model
    schema = st.session_state.feature_schema
    problem_type = st.session_state.problem_type

    mode = st.radio("Prediction Mode", ["Single", "Batch CSV"])

    if mode == "Single":

        # ── FRIENDLY INPUT UI ──────────────────────────────
        # Instead of showing encoded columns like Ship_Mode_Same_Day,
        # show clean dropdowns for categorical features
        # and number inputs only for real numeric features

        inputs = {}

        # Core numeric features — always show these
        core_numeric = ["Sales", "Quantity", "Discount"]
        for col in core_numeric:
            if col in schema:
                inputs[col] = st.number_input(col, value=0.0)

        # Friendly categorical dropdowns
        cat_options = {
            "Ship Mode": ["First Class", "Same Day", "Second Class", "Standard Class"],
            "Segment":   ["Consumer", "Corporate", "Home Office"],
            "Region":    ["Central", "East", "South", "West"],
            "Category":  ["Furniture", "Office Supplies", "Technology"],
            "Sub-Category": [
                "Accessories", "Appliances", "Art", "Binders", "Bookcases",
                "Chairs", "Copiers", "Envelopes", "Fasteners", "Furnishings",
                "Labels", "Machines", "Paper", "Phones", "Storage",
                "Supplies", "Tables"
            ]
        }

        selected_cats = {}
        for friendly_name, options in cat_options.items():
            # Check if any encoded version exists in schema
            col_prefix = friendly_name.replace(" ", "_").replace("-", "_") + "_"
            if any(c.startswith(col_prefix) for c in schema):
                selected_cats[friendly_name] = st.selectbox(friendly_name, options)

        # Smart date input — only show if datetime cols exist in training data
        date = None
        has_date_features = any(
            "_year" in c or "_month" in c or "_dayofweek" in c
            for c in schema
        )

        if has_date_features:
            date = st.date_input("📅 Order Date (used for seasonality features)")
        else:
            # Use today as default silently — won't affect prediction
            from datetime import date as dt
            date = dt.today()

        # Build encoded inputs from friendly selections
        for friendly_name, selected_val in selected_cats.items():
            col_prefix = friendly_name.replace(" ", "_").replace("-", "_") + "_"
            for col in schema:
                if col.startswith(col_prefix):
                    suffix = col[len(col_prefix):]
                    # Match the selected value to encoded column
                    encoded_val = selected_val.replace(" ", "_").replace("-", "_")
                    inputs[col] = 1 if suffix == selected_val or suffix == encoded_val else 0

        # Date features
        for col in schema:
            if "_year" in col:   inputs[col] = date.year
            elif "_month" in col: inputs[col] = date.month
            elif "_day" in col and "_dayofweek" not in col and "_is_weekend" not in col:
                inputs[col] = date.day
            elif "_dayofweek" in col: inputs[col] = date.weekday()
            elif "_is_weekend" in col: inputs[col] = int(date.weekday() >= 5)

        # Fill any remaining schema columns with 0
        for col in schema:
            if col not in inputs:
                inputs[col] = 0

        if st.button("Predict"):
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
                st.info(f"Range: {pred-1.5*std:.2f} – {pred+1.5*std:.2f}")

                # AI explanation of prediction
                if api_key:
                    with st.spinner("AI explaining this prediction..."):
                        explanation = chat_with_data(
                            api_key,
                            f"The model predicted {pred:.2f} for {st.session_state.target_col}. In 2 sentences, explain what this means for the business and what the main drivers likely are.",
                            [],
                            st.session_state.model_card,
                            profile, df,
                            problem_type,
                            st.session_state.target_col,
                            st.session_state.business_insights or []
                        )
                    st.info(f"🤖 {explanation}")
            else:
                prob = model.predict_proba(Xp)[0]
                st.success(f"Class: {np.argmax(prob)}")
                st.info(f"Confidence: {np.max(prob)*100:.1f}%")

    else:
        batch = st.file_uploader("Upload CSV", type=["csv"])
        if batch:
            batch_df = load_cached_csv(batch)
            Xb = prepare_features(batch_df, profile, training=False, feature_schema=schema)
            preds = model.predict(Xb)
            out = batch_df.copy()
            out["prediction"] = preds
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button(
                "Download CSV",
                out.to_csv(index=False).encode(),
                "batch_predictions.csv"
            )

# ======================================================
# 🧪 EXPERIMENT HISTORY
# ======================================================
elif page == "🧪 Experiment History":
    exp = load_experiments()
    if exp:
        st.dataframe(pd.DataFrame(exp), use_container_width=True)
    else:
        st.info("No experiments yet.")

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
        # Enhance report with AI narrative if available
        insights_for_report = st.session_state.business_insights.copy()
        if st.session_state.agent_narrative:
            insights_for_report = [st.session_state.agent_narrative] + insights_for_report

        path = generate_pdf_report(
            st.session_state.model_card,
            insights_for_report
        )
        with open(path, "rb") as f:
            st.download_button("Download PDF Report", f, "DataPilot_Report.pdf")
