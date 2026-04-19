import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
    "X": None, "y": None, "model": None,
    "problem_type": None, "target_col": None,
    "business_insights": None, "analyzed": False,
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
    if uploaded.size > 5 * 1024 * 1024:
        st.error("File too large (max 5MB)")
        st.stop()

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
# 🚀 ONBOARDING
# ======================================================
if not st.session_state.analyzed:
    st.markdown("## 🚀 Welcome to DataAgentX")

    st.markdown("""
### 💡 What this does:
Upload your business data → get AI insights → take action

### ⚡ In a few minutes you will:
✔ Understand what drives your revenue  
✔ Detect risks in your business  
✔ Get clear actions to improve  

### 📂 What data works best:
- Shopify exports  
- Sales / revenue CSV  
- Order-level data (better insights)
""")

    st.info("👉 Click **Analyze** after uploading your data")
    st.divider()

# ======================================================
# 🧠 DATA DETECTION
# ======================================================
columns_str = " ".join(df.columns).lower()
has_customer = "customer" in columns_str

st.sidebar.markdown("### 🧠 Data Detection")
if has_customer:
    st.sidebar.success("👤 Customer-level data")
else:
    st.sidebar.warning("📊 Aggregated data (no churn available)")

# ======================================================
# MODE
# ======================================================
st.subheader("🎯 Choose Analysis Type")

options = ["🧠 CEO Dashboard", "📊 Revenue Analysis"]
if has_customer:
    options.append("👤 Churn Analysis")
options.append("🔢 Custom")

mode = st.selectbox("Analysis Mode", options)

# ======================================================
# 🧠 CEO DASHBOARD
# ======================================================
if mode == "🧠 CEO Dashboard":

    revenue_df = adaptive_preprocess(df, "revenue")

    try:
        if has_customer:
            churn_df = adaptive_preprocess(df, "churn")
            churn_available = True
        else:
            raise ValueError()
    except:
        churn_available = False
        churn_df = None

    total_revenue = revenue_df["Revenue"].sum()
    total_orders = revenue_df["Orders"].sum()
    aov = total_revenue / max(total_orders, 1)

    churn_rate = churn_df["churn"].mean() if churn_available else None

    st.markdown("## 🧠 CEO Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"${total_revenue:,.0f}")
    c2.metric("Orders", int(total_orders))
    c3.metric("Churn Rate", f"{churn_rate*100:.1f}%" if churn_rate is not None else "N/A")
    c4.metric("AOV", f"${aov:.2f}")

    st.divider()

    if churn_rate is not None and churn_rate > 0.4:
        st.error("⚠️ High churn is hurting growth")
    elif total_revenue < revenue_df["Revenue"].median():
        st.warning("📉 Revenue unstable")
    else:
        st.success("🚀 Business healthy")

    st.markdown("### 💡 CEO Actions")

    if churn_available:
        st.success("""
👉 Reduce churn  
👉 Improve retention campaigns  
👉 Focus on repeat customers  
""")
    else:
        st.info("""
👉 Upload customer-level data for churn insights  
👉 Focus on increasing orders  
👉 Optimize pricing  
""")

    st.stop()

# ======================================================
# PREPROCESS
# ======================================================
if mode == "📊 Revenue Analysis":
    processed_df = adaptive_preprocess(df, "revenue")
    target = "Revenue"

elif mode == "👤 Churn Analysis":

    if not has_customer:
        st.error("❌ Churn requires customer data")
        st.stop()

    processed_df = adaptive_preprocess(df, "churn")
    target = "churn"

else:
    processed_df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Select Target", numeric_cols)

st.session_state.target_col = target

# ======================================================
# ANALYZE
# ======================================================
if st.button("🚀 Analyze"):

    X, y = prepare_features(processed_df, profile, target)
    X = X.select_dtypes(include=np.number).fillna(0)

    problem = detect_problem_type(y)
    results, best_model, metrics = train_models(X, y, problem)
    model = joblib.load("models/best_model.pkl")

    try:
        sample_X = X.sample(min(500, len(X)))
        explainer = shap.Explainer(model, sample_X)
        sv = explainer(sample_X)

        insights = generate_business_impact(
            sv.values, sample_X, problem, target
        )
    except:
        insights = ["Basic insights available"]

    st.session_state.update({
        "X": X,
        "y": y,
        "model": model,
        "problem_type": problem,
        "business_insights": insights,
        "analyzed": True,
        "model_card": {
            "model": best_model,
            "features": X.shape[1],
            "metrics": metrics
        },
        "processed_df": processed_df
    })

    st.success("✅ Analysis complete")

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
# 🚨 ALERT SYSTEM
# ======================================================
if st.session_state.analyzed:

    st.subheader("🚨 Smart Alerts")

    df_disp = st.session_state.processed_df
    alerts = []

    if "Revenue" in df_disp.columns:
        recent = df_disp["Revenue"].tail(7).mean()
        overall = df_disp["Revenue"].mean()

        if recent < 0.8 * overall:
            alerts.append("📉 Revenue dropped significantly in recent days")

        if df_disp["Revenue"].std() > 0.5 * df_disp["Revenue"].mean():
            alerts.append("⚠️ Revenue is highly unstable")

    if "churn" in df_disp.columns:
        churn_rate = df_disp["churn"].mean()
        if churn_rate > 0.4:
            alerts.append("🚨 High customer churn detected")

    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ No major risks detected")

# ======================================================
# 🔮 FUTURE OUTLOOK
# ======================================================
if st.session_state.analyzed and target.lower() == "revenue":

    st.subheader("🔮 Future Outlook")

    df_disp = st.session_state.processed_df

    recent_avg = df_disp["Revenue"].tail(7).mean()
    overall_avg = df_disp["Revenue"].mean()

    change_pct = ((recent_avg - overall_avg) / max(overall_avg, 1)) * 100

    if change_pct > 10:
        st.success(f"📈 Revenue trending UP (+{change_pct:.1f}%)")
    elif change_pct < -10:
        st.error(f"📉 Revenue trending DOWN ({change_pct:.1f}%)")
    else:
        st.info("➡️ Revenue is stable")

# ======================================================
# 💬 CHAT
# ======================================================
st.markdown("### ⚡ Business Decisions")

col1, col2, col3 = st.columns(3)

if col1.button("📉 Why is revenue dropping?"):
    st.session_state.pending_question = "Why is revenue dropping and what should I fix immediately?"

elif col2.button("💰 How to increase revenue fast?"):
    st.session_state.pending_question = "What actions will quickly increase revenue?"

elif col3.button("⚠️ Biggest risk right now?"):
    st.session_state.pending_question = "What is the biggest business risk right now?"

col4, col5, col6 = st.columns(3)

if col4.button("🎯 Where to focus?"):
    st.session_state.pending_question = "Where should I focus for maximum impact?"

elif col5.button("📈 Growth strategy"):
    st.session_state.pending_question = "Give me a growth strategy based on this data"

elif col6.button("🔥 Immediate actions"):
    st.session_state.pending_question = "What should I do TODAY to improve results?"

st.subheader("💬 Ask AI")

if st.session_state.analyzed:

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about your business")

    if user_input:
        st.session_state.pending_question = user_input

    if st.session_state.pending_question:

        q = st.session_state.pending_question
        st.session_state.pending_question = None

        st.session_state.chat_history.append({"role": "user", "content": q})

        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_data(
                    api_key,
                    q,
                    st.session_state.chat_history[:-1],
                    st.session_state.model_card,
                    profile,
                    st.session_state.processed_df,
                    st.session_state.problem_type,
                    target,
                    st.session_state.business_insights
                )
                st.write(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

# ======================================================
# ⚙️ TECHNICAL VIEW
# ======================================================
with st.expander("⚙️ Technical Details"):

    st.markdown("### 🧠 Pipeline")
    st.code("Upload → Preprocess → Feature Engineering → AutoML → SHAP → Insights → Chat")

    st.markdown("### 🤖 Model Info")
    st.write(st.session_state.model_card)
