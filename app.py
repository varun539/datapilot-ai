import streamlit as st
import pandas as pd
import numpy as np
import shap
import os

from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data
from src.eda import basic_profile
from src.adaptive_preprocess import adaptive_preprocess

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataAgentX", layout="wide")
api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")

# ======================================================
# SESSION STATE
# ======================================================
DEFAULTS = {
    "df": None, "profile": None,
    "model": None,
    "problem_type": None, "target_col": None,
    "business_insights": [], "analyzed": False,
    "chat_history": [], "model_card": {},
    "processed_df": None,
    "pending_question": None
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🚀 DataAgentX")

uploaded = st.sidebar.file_uploader("Upload CSV")

if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state.df = df
    st.session_state.profile = basic_profile(df)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

if st.sidebar.button("🎯 Load Demo"):
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
    st.session_state.df = df
    st.session_state.profile = basic_profile(df)
    st.session_state.analyzed = False
    st.session_state.chat_history = []

df = st.session_state.df
profile = st.session_state.profile

if df is None:
    st.info("👈 Upload dataset to start")
    st.stop()

# ======================================================
# ONBOARDING
# ======================================================
if not st.session_state.analyzed:
    st.markdown("## 🚀 Welcome to DataAgentX")

    st.markdown("""
Upload your business data → get AI insights → take action

✔ Understand drivers  
✔ Detect risks  
✔ Get actions  
""")

    st.info("👉 Click Analyze")
    st.divider()

# ======================================================
# MODE
# ======================================================
st.subheader("🎯 Choose Analysis Type")

columns_str = " ".join(df.columns).lower()
has_customer = "customer" in columns_str

options = ["📊 Revenue Analysis"]
if has_customer:
    options.append("👤 Churn Analysis")
options.append("🔢 Custom")

mode = st.selectbox("Mode", options)

# ======================================================
# PREPROCESS
# ======================================================
try:
    if mode == "📊 Revenue Analysis":
        processed_df = adaptive_preprocess(df, "revenue")
        target = "Revenue"

    elif mode == "👤 Churn Analysis":
        processed_df = adaptive_preprocess(df, "churn")
        target = "churn"

    else:
        processed_df = df.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target = st.selectbox("Target", numeric_cols)

except:
    processed_df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = numeric_cols[0]

st.session_state.target_col = target

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Analyze"):

    try:
        X, y = prepare_features(processed_df, profile, target)
        X = X.select_dtypes(include=np.number).fillna(0)

        problem = detect_problem_type(y)

        # ✅ FIX: use model directly (NO joblib)
        results, model, metrics = train_models(X, y, problem)

        # SHAP
        try:
            sample_X = X.sample(min(300, len(X)))
            explainer = shap.Explainer(model, sample_X)
            sv = explainer(sample_X)

            insights = generate_business_impact(
                sv.values, sample_X, problem, target
            )
        except:
            insights = ["Basic insights available"]

        st.session_state.update({
            "model": model,
            "problem_type": problem,
            "business_insights": insights,
            "analyzed": True,
            "processed_df": processed_df,
            "model_card": {
                "model": type(model).__name__,
                "features": X.shape[1],
                "metrics": metrics
            }
        })

        st.success("✅ Analysis complete")

    except Exception as e:
        st.error(f"Analysis failed: {e}")

# ======================================================
# RESULTS
# ======================================================
if st.session_state.analyzed:

    st.subheader("🏆 Model Summary")
    st.write(st.session_state.model_card)

    st.subheader("📊 Insights")
    for i in st.session_state.business_insights:
        st.info(i)

    # ======================================================
    # ALERTS
    # ======================================================
    st.subheader("🚨 Smart Alerts")

    df_disp = st.session_state.processed_df if st.session_state.processed_df is not None else df

    alerts = []

    if target in df_disp.columns:
        vals = df_disp[target].dropna()

        if len(vals) > 10:
            if vals.tail(5).mean() < vals.mean():
                alerts.append("📉 Trend declining")

            if vals.std() > 0.5 * vals.mean():
                alerts.append("⚠️ High volatility")

    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ No major risks")

# ======================================================
# CHAT
# # ==========================================


# ======================================================
# 💬 CHAT (FINAL FIX)
# ======================================================
st.subheader("💬 Ask AI")

if st.session_state.analyzed:

    # INIT
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # SHOW HISTORY FIRST
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # USER INPUT
    user_input = st.chat_input("Ask about your business")

    # QUICK BUTTONS (NO pending_question anymore)
    st.markdown("### ⚡ Business Decisions")

    col1, col2, col3 = st.columns(3)

    if col1.button("📉 Why is revenue dropping?"):
        user_input = "Why is revenue dropping and what should I fix immediately?"

    elif col2.button("💰 How to increase revenue fast?"):
        user_input = "What actions will quickly increase revenue?"

    elif col3.button("⚠️ Biggest risk right now?"):
        user_input = "What is the biggest business risk right now?"

    col4, col5, col6 = st.columns(3)

    if col4.button("🎯 Where to focus?"):
        user_input = "Where should I focus for maximum impact?"

    elif col5.button("📈 Growth strategy"):
        user_input = "Give me a growth strategy based on this data"

    elif col6.button("🔥 Immediate actions"):
        user_input = "What should I do TODAY to improve results?"

    # PROCESS INPUT
    if user_input:

        # ADD USER
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # AI RESPONSE
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_data(
                    api_key,
                    user_input,
                    st.session_state.chat_history[:-1],  # IMPORTANT
                    st.session_state.model_card,
                    profile,
                    st.session_state.processed_df,
                    st.session_state.problem_type,
                    st.session_state.target_col,
                    st.session_state.business_insights
                )
                st.write(response)

        # SAVE RESPONSE
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })




# ======================================================
# TECHNICAL
# ======================================================
with st.expander("⚙️ Technical Details"):

    st.code("Upload → Preprocess → AutoML → SHAP → Insights")

    st.write(st.session_state.model_card)
