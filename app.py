import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

from src.pipeline import prepare_features
from src.automl import detect_problem_type, train_models
from src.impact import generate_business_impact
from src.agent import chat_with_data, suggest_target_column, generate_agent_narrative
from src.report import generate_pdf_report
from src.eda import basic_profile

st.set_page_config(page_title="DataAgentX", layout="wide")

api_key = os.getenv("OPENAI_API_KEY")

# =========================
# SESSION STATE
# =========================
for key in ["df", "profile", "X", "y", "model", "problem_type", "target_col", "business_insights"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================
# FILE + DEMO
# =========================
st.sidebar.header("📂 Upload Data")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("### 🎯 Try Demo")

if st.sidebar.button("Use Demo Dataset"):
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Walmart.csv")
    profile = basic_profile(df)
    st.session_state["df"] = df
    st.session_state["profile"] = profile

if file:
    df = pd.read_csv(file)
    profile = basic_profile(df)
    st.session_state["df"] = df
    st.session_state["profile"] = profile

df = st.session_state.get("df")
profile = st.session_state.get("profile")

# =========================
# FUNCTIONS
# =========================
def generate_executive_summary(df, target):
    if df is None or target not in df.columns:
        return {}

    vals = df[target].dropna()
    if len(vals) < 10:
        return {}

    recent = vals.tail(int(len(vals)*0.2)).mean()
    prev = vals.head(int(len(vals)*0.8)).mean()

    change = (recent - prev) / (prev + 1e-6)

    slope = np.polyfit(range(len(vals)), vals, 1)[0]

    trend = "Growing" if slope > 0 else "Declining"
    risk = "High" if change < -0.1 else "Medium" if change < -0.05 else "Low"

    return {"trend": trend, "change": change, "risk": risk}


def generate_alerts(df, target):
    if df is None or target not in df.columns:
        return []

    vals = df[target].dropna()
    if len(vals) < 20:
        return []

    recent = vals.tail(int(len(vals)*0.2)).mean()
    prev = vals.head(int(len(vals)*0.8)).mean()

    change = (recent - prev) / (prev + 1e-6)

    alerts = []

    if change < -0.1:
        alerts.append({
            "msg": f"⚠️ {target} dropped by {abs(change)*100:.1f}%",
            "action": "Increase promotions or adjust pricing."
        })
    elif change > 0.1:
        alerts.append({
            "msg": f"📈 {target} increased by {abs(change)*100:.1f}%",
            "action": "Increase inventory or marketing."
        })

    return alerts

# =========================
# UI
# =========================
st.title("🚀 DataAgentX")

if df is None:
    st.info("Upload dataset or use demo")
    st.stop()

st.subheader("📊 Dashboard")

target = suggest_target_column(api_key, df.columns.tolist(), df)
st.info(f"🎯 Target: {target}")

# =========================
# EXEC SUMMARY
# =========================
summary = generate_executive_summary(df, target)

if summary:
    col1, col2, col3 = st.columns(3)
    col1.metric("Trend", summary["trend"])
    col2.metric("Change", f"{summary['change']*100:.1f}%")
    col3.metric("Risk", summary["risk"])

# =========================
# ALERTS
# =========================
alerts = generate_alerts(df, target)

for a in alerts:
    st.warning(a["msg"])
    st.success(f"👉 {a['action']}")

# =========================
# ANALYSIS
# =========================
if st.button("🚀 Analyze"):

    X = prepare_features(df, profile, target, training=True)
    y = pd.to_numeric(df[target], errors="coerce").fillna(df[target].median())

    problem = detect_problem_type(y)
    results, best_model = train_models(X, y, problem)

    model = joblib.load("models/best_model.pkl")

    st.success(f"Best Model: {best_model}")
    st.dataframe(results)

    # SHAP
    try:
        explainer = shap.Explainer(model, X)
        shap_vals = explainer(X)

        insights = generate_business_impact(shap_vals.values, X, problem, target)

        for i in insights:
            st.info(i)

    except:
        st.warning("SHAP failed")

# =========================
# CHAT
# =========================
st.subheader("💬 Ask AI")

q = st.text_input("Ask about your data")

if q:
    try:
        res = chat_with_data(api_key, q, [], {}, {}, df, None, target, [])
        st.success(res)
    except:
        st.warning("Chat failed")

# =========================
# REPORT
# =========================
if st.button("📄 Generate Report"):
    path = generate_pdf_report({}, [])
    with open(path, "rb") as f:
        st.download_button("Download Report", f)
