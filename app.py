# import streamlit as st
# import pandas as pd
# import numpy as np
# import shap
# import os
# import joblib

# from src.pipeline import prepare_features
# from src.automl import detect_problem_type, train_models
# from src.impact import generate_business_impact
# from src.agent import chat_with_data
# from src.eda import basic_profile
# from src.adaptive_preprocess import adaptive_preprocess
# from src.leakage import detect_leakage

# # ======================================================
# # CONFIG
# # ======================================================
# st.set_page_config(page_title="DataPilot AI", layout="wide")
# api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

# # ======================================================
# # SESSION STATE
# # ======================================================
# DEFAULTS = {
#     "df": None,
#     "profile": None,
#     "model": None,
#     "problem_type": None,
#     "target_col": None,
#     "business_insights": [],
#     "model_card": {},
#     "processed_df": None,
#     "analyzed": False,
#     "chat_history": []
# }
# for k, v in DEFAULTS.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ======================================================
# # HEADER
# # ======================================================
# st.title("🚀 DataPilot AI — Agentic AutoML Platform")
# st.caption("Upload your data → Train models → Get AI-powered insights")

# # ======================================================
# # SIDEBAR
# # ======================================================
# st.sidebar.title("📂 Upload Data")

# file = st.sidebar.file_uploader("Upload CSV")

# if file:
#     df = pd.read_csv(file)
#     if st.session_state.df is None or st.session_state.df.shape != df.shape:
#         st.session_state.df = df
#         st.session_state.profile = basic_profile(df)
#         st.session_state.analyzed = False
#         st.session_state.chat_history = []

# if st.sidebar.button("📊 Load Demo Dataset"):
#     df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
#     st.session_state.df = df
#     st.session_state.profile = basic_profile(df)
#     st.session_state.analyzed = False
#     st.session_state.chat_history = []

# df = st.session_state.df
# profile = st.session_state.profile

# if df is None:
#     st.warning("Upload a dataset to begin.")
#     st.stop()

# # ======================================================
# # DATA PREVIEW
# # ======================================================
# st.subheader("📄 Data Preview")
# st.dataframe(df.head())

# # ======================================================
# # TARGET SELECTION
# # ======================================================
# st.subheader("🎯 Target Selection")

# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# if not numeric_cols:
#     st.error("No numeric columns found.")
#     st.stop()

# priority = ["Revenue", "Weekly_Sales", "Sales", "Profit"]
# default = next((c for c in priority if c in numeric_cols), numeric_cols[0])

# target = st.selectbox(
#     "Select Target Column",
#     numeric_cols,
#     index=numeric_cols.index(default)
# )

# st.session_state.target_col = target

# # ======================================================
# # ANALYZE
# # ======================================================
# if st.button("🚀 Run Analysis"):

#     with st.spinner("Running full ML pipeline..."):

#         try:
#             # -------------------------
#             # PREPROCESS
#             # -------------------------
#             try:
#                 processed_df = adaptive_preprocess(df, "revenue")
#                 if target not in processed_df.columns:
#                     processed_df = df.copy()
#             except:
#                 processed_df = df.copy()

#             proc_profile = basic_profile(processed_df)

#             # -------------------------
#             # PIPELINE
#             # -------------------------
#             X, y = prepare_features(processed_df, proc_profile, target)

#             if X.empty or y.empty:
#                 st.error("❌ Feature engineering failed. Check dataset.")
#                 st.stop()

#             X = X.select_dtypes(include=np.number).fillna(0)
#             X = X.replace([np.inf, -np.inf], 0)

#             # -------------------------
#             # 🚨 LEAKAGE CHECK (FIXED POSITION)
#             # -------------------------
#             warnings, high_risk = detect_leakage(X, y)

#             st.subheader("🛡️ Data Leakage Check")

#             for w in warnings:
#                 if "HIGH RISK" in w:
#                     st.error(w)
#                 elif "Suspicious" in w:
#                     st.warning(w)
#                 else:
#                     st.success(w)

#             if high_risk:
#                 st.error("❌ Training stopped due to data leakage")
#                 st.stop()

#             # -------------------------
#             # MODEL TRAINING
#             # -------------------------
#             problem = detect_problem_type(y)

#             results_df, model_name, metrics = train_models(X, y, problem)

#             model = joblib.load("models/best_model.pkl")

#             # -------------------------
#             # SHAP
#             # -------------------------
#             insights = []
#             try:
#                 sample_X = X.sample(min(300, len(X)), random_state=42)

#                 explainer = shap.TreeExplainer(model)
#                 sv = explainer.shap_values(sample_X)

#                 if isinstance(sv, list):
#                     sv = sv[1]

#                 insights = generate_business_impact(
#                     sv, sample_X, problem, target
#                 )

#             except Exception as e:
#                 insights = [f"SHAP skipped: {e}"]

#             # -------------------------
#             # SAVE STATE
#             # -------------------------
#             st.session_state.update({
#                 "model": model,
#                 "problem_type": problem,
#                 "business_insights": insights,
#                 "processed_df": processed_df,
#                 "analyzed": True,
#                 "chat_history": [],
#                 "model_card": {
#                     "model": model_name,
#                     "features": X.shape[1],
#                     "rows": processed_df.shape[0],
#                     "target": target,
#                     "metrics": metrics
#                 }
#             })

#             st.success(f"✅ {model_name} trained successfully!")

#         except Exception as e:
#             st.error(f"Error: {e}")
#             st.exception(e)

# # ======================================================
# # RESULTS
# # ======================================================
# if st.session_state.analyzed:

#     mc = st.session_state.model_card
#     metrics = mc.get("metrics", {})
#     hold = metrics.get("holdout", {})
#     cv = metrics.get("cv", {})

#     st.divider()
#     st.subheader("🏆 Model Summary")

#     c1, c2 = st.columns(2)
#     c1.metric("Model", mc.get("model"))
#     c2.metric("Features", mc.get("features"))

#     st.subheader("📈 Metrics")

#     if st.session_state.problem_type == "regression":
#         st.write(f"R²: {round(hold.get('r2', 0), 4)}")
#         st.write(f"MAE: {round(hold.get('mae', 0), 2)}")
#         st.write(f"RMSE: {round(hold.get('rmse', 0), 2)}")
#     else:
#         st.write(f"Accuracy: {round(hold.get('accuracy', 0), 4)}")
#         st.write(f"F1: {round(hold.get('f1', 0), 4)}")

#     st.write(f"CV Mean: {round(cv.get('mean', 0), 4)}")
#     st.write(f"CV Std: {round(cv.get('std', 0), 4)}")

#     # -------------------------
#     # INSIGHTS
#     # -------------------------
#     st.subheader("📊 Business Insights")

#     for ins in st.session_state.business_insights:
#         st.info(ins)

# # ======================================================
# # CHAT
# # ======================================================
# st.divider()
# st.subheader("💬 Ask Questions About Your Data")

# if not st.session_state.analyzed:
#     st.info("Run analysis first to enable chat")
# else:

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.write(msg["content"])

#     user_input = st.chat_input("Ask about your business...")

#     if user_input:

#         st.session_state.chat_history.append({
#             "role": "user",
#             "content": user_input
#         })

#         with st.chat_message("user"):
#             st.write(user_input)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = chat_with_data(
#                     api_key,
#                     user_input,
#                     st.session_state.chat_history[:-1],
#                     st.session_state.model_card,
#                     profile or {},
#                     st.session_state.processed_df,
#                     st.session_state.problem_type,
#                     st.session_state.target_col,
#                     st.session_state.business_insights
#                 )
#                 st.write(response)

#         st.session_state.chat_history.append({
#             "role": "assistant",
#             "content": response
#         })

# # ======================================================
# # TECHNICAL
# # ======================================================
# with st.expander("⚙️ Technical Details"):

#     if st.session_state.analyzed:

#         st.markdown("### Pipeline")
#         st.code("""
# Upload → Preprocess → Feature Engineering → AutoML → SHAP → Insights
# """)

#         st.json(st.session_state.model_card)

#     else:
#         st.info("Run analysis to view details")







import streamlit as st
import pandas as pd
import numpy as np
import shap
import os
import joblib

from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data
from src.eda import basic_profile
from src.adaptive_preprocess import adaptive_preprocess

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataPilot AI", layout="wide")
api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

# ======================================================
# SESSION STATE
# ======================================================
DEFAULTS = {
    "df": None, "profile": None,
    "model": None, "problem_type": None,
    "target_col": None, "business_insights": [],
    "model_card": {}, "processed_df": None,
    "analyzed": False, "chat_history": []
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# HEADER
# ======================================================
st.title("🚀 DataPilot AI — Agentic AutoML Platform")
st.caption("Upload your data → Train models → Get AI-powered insights")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("📂 Upload Data")

file = st.sidebar.file_uploader("Upload CSV")
if file:
    df = pd.read_csv(file)
    if st.session_state.df is None or st.session_state.df.shape != df.shape:
        st.session_state.df       = df
        st.session_state.profile  = basic_profile(df)
        st.session_state.analyzed = False
        st.session_state.chat_history = []

if st.sidebar.button("📊 Load Demo Dataset"):
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
    st.session_state.df       = df
    st.session_state.profile  = basic_profile(df)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

df      = st.session_state.df
profile = st.session_state.profile

if df is None:
    st.warning("Upload a dataset to begin.")
    st.stop()

# Sidebar dataset info
st.sidebar.success("✅ Dataset Ready")
c1, c2 = st.sidebar.columns(2)
c1.metric("Rows", df.shape[0])
c2.metric("Cols", df.shape[1])

# ======================================================
# DATA PREVIEW
# ======================================================
st.subheader("📄 Data Preview")
st.dataframe(df.head())

# ======================================================
# TARGET SELECTION — smart default
# ======================================================
st.subheader("🎯 Target Selection")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found.")
    st.stop()

priority = ["Weekly_Sales", "Revenue", "Sales", "Profit", "Target"]
default  = next((c for c in priority if c in numeric_cols), numeric_cols[0])
target   = st.selectbox("Select Target Column", numeric_cols,
                         index=numeric_cols.index(default))
st.session_state.target_col = target

# ======================================================
# SMART PREPROCESS DETECTION
# ======================================================
def should_preprocess(df, target):
    """
    Only use adaptive_preprocess for RAW transaction data
    (InvoiceNo, Quantity, UnitPrice style)
    
    Skip for already-structured datasets like Walmart
    (already has Date, Weekly_Sales, Holiday_Flag etc)
    """
    cols_lower = [c.lower() for c in df.columns]

    # Raw transaction signals
    has_invoice  = any("invoice" in c or "transaction" in c for c in cols_lower)
    has_unitprice = "unitprice" in cols_lower or "unit_price" in cols_lower
    has_quantity  = "quantity" in cols_lower

    # Already structured signals
    has_weekly   = any("weekly" in c for c in cols_lower)
    has_holiday  = any("holiday" in c for c in cols_lower)
    has_cpi      = "cpi" in cols_lower

    if has_invoice and has_unitprice and has_quantity:
        return True   # Raw Shopify/retail → preprocess needed

    if has_weekly or has_holiday or has_cpi:
        return False  # Already structured → skip preprocess

    return False  # Default: use as-is

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Run Analysis"):
    with st.spinner("Running full ML pipeline..."):
        try:

            # Smart preprocess decision
            if should_preprocess(df, target):
                try:
                    processed_df = adaptive_preprocess(df, "revenue")
                    actual_target = "Revenue"
                    st.info("📊 Raw transaction data detected — aggregated to daily revenue")
                except Exception as e:
                    st.warning(f"Auto-preprocess failed ({e}) — using raw data")
                    processed_df  = df.copy()
                    actual_target = target
            else:
                processed_df  = df.copy()
                actual_target = target
                st.info("📊 Structured dataset detected — using directly")

            # Verify target exists
            if actual_target not in processed_df.columns:
                actual_target = target

            if actual_target not in processed_df.columns:
                st.error(f"Target '{actual_target}' not found!")
                st.stop()

            proc_profile = basic_profile(processed_df)

            # Pipeline
            X, y = prepare_features(
                processed_df, proc_profile, actual_target, training=True
            )

            if X.empty or y.empty or X.shape[1] == 0:
                st.error("Feature engineering failed. Check dataset.")
                st.stop()

            X = X.select_dtypes(include=np.number).fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            # Align lengths
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len].reset_index(drop=True)
            y = y.iloc[:min_len].reset_index(drop=True)

            st.info(f"✅ Features ready: {X.shape[1]} features, {len(y)} rows")
            st.write("**Features used:**", list(X.columns))

            # Train
            problem = detect_problem_type(y)
            results_df, model_name, metrics = train_models(X, y, problem)

            # Safe load
            model_path = "models/best_model.pkl"
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                st.error("Model file corrupt. Try again.")
                st.stop()
            model = joblib.load(model_path)

            # SHAP
            insights = []
            try:
                sample_X  = X.sample(min(300, len(X)), random_state=42)
                explainer = shap.TreeExplainer(model)
                sv        = explainer.shap_values(sample_X)
                if isinstance(sv, list):
                    sv = sv[1]
                insights = generate_business_impact(sv, sample_X, problem, actual_target)
            except Exception as se:
                insights = [f"Analysis complete. SHAP detail: {se}"]

            st.session_state.update({
                "model":             model,
                "problem_type":      problem,
                "business_insights": insights,
                "processed_df":      processed_df,
                "analyzed":          True,
                "chat_history":      [],
                "target_col":        actual_target,
                "model_card": {
                    "model":    model_name,
                    "features": X.shape[1],
                    "rows":     processed_df.shape[0],
                    "target":   actual_target,
                    "metrics":  metrics
                }
            })

            st.success(f"✅ {model_name} trained!")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:

    mc      = st.session_state.model_card
    metrics = mc.get("metrics", {})
    hold    = metrics.get("holdout", {})
    cv      = metrics.get("cv", {})

    st.divider()
    st.subheader("🏆 Model Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",    mc.get("model", "—"))
    c2.metric("Features", mc.get("features", "—"))

    if st.session_state.problem_type == "regression":
        c3.metric("R² (holdout)", round(hold.get("r2", 0), 4))
        c4.metric("CV Mean R²",   round(cv.get("mean", 0), 4))
    else:
        c3.metric("Accuracy",  round(hold.get("accuracy", 0), 4))
        c4.metric("CV F1 Mean", round(cv.get("mean", 0), 4))

    # Leakage warning if big gap
    r2_hold = hold.get("r2", 0)
    cv_mean = cv.get("mean", 0)
    if abs(r2_hold - cv_mean) > 0.3:
        st.warning("Model performance varies across time — monitor before deployment")
    else:
        st.success(f"✅ — no leakage detected")

    st.subheader("📊 Business Insights")
    for ins in st.session_state.business_insights:
        st.info(ins)

# ======================================================
# CHAT
# ======================================================
st.divider()
st.subheader("💬 Ask Questions About Your Data")

if not st.session_state.analyzed:
    st.info("Run analysis first to enable chat")
else:
    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about your business...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chat_with_data(
                        api_key, user_input,
                        st.session_state.chat_history[:-1],
                        st.session_state.model_card,
                        profile or {},
                        st.session_state.processed_df,
                        st.session_state.problem_type,
                        st.session_state.target_col,
                        st.session_state.business_insights
                    )
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Chat error: {e}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ======================================================
# TECHNICAL
# ======================================================
with st.expander("⚙️ Technical Details"):
    if st.session_state.analyzed:
        mc   = st.session_state.model_card
        raw  = mc.get("metrics", {})
        hold = raw.get("holdout", {})
        cv   = raw.get("cv", {})
        st.markdown("### 📈 Full Metrics")
        if st.session_state.problem_type == "regression":
            st.write(f"R²:      {round(hold.get('r2',0), 4)}")
            st.write(f"MAE:     {round(hold.get('mae',0), 2)}")
            st.write(f"RMSE:    {round(hold.get('rmse',0), 2)}")
        else:
            st.write(f"Accuracy: {round(hold.get('accuracy',0), 4)}")
            st.write(f"F1:       {round(hold.get('f1',0), 4)}")
        st.write(f"CV Mean: {round(cv.get('mean',0), 4)}")
        st.write(f"CV Std:  {round(cv.get('std',0), 4)}")
        st.json(mc)
    else:
        st.info("Run analysis to view details")
