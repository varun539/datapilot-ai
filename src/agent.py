# src/agent.py

from openai import OpenAI
import json
import numpy as np
import pandas as pd


# ======================================================
# INIT CLIENT
# ======================================================
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


# ======================================================
# CORE: GENERATE AGENTIC NARRATIVE
# ======================================================
def generate_agent_narrative(
    api_key: str,
    model_card: dict,
    business_insights: list,
    profile: dict,
    shap_top_features: list,
    problem_type: str,
    target_col: str
) -> str:

    client = get_client(api_key)

    # ✅ SAFETY FIX
    model_card = model_card or {}
    business_insights = business_insights or []
    shap_top_features = shap_top_features or []

    context = f"""
You are a senior data scientist reviewing an AutoML analysis.

MODEL CARD:
- Model: {model_card.get('model')}
- Problem Type: {problem_type}
- Target: {target_col}
- Rows: {model_card.get('rows')}
- Features: {model_card.get('features')}
- Performance: {json.dumps(model_card.get('performance', {}))}

TOP SHAP FEATURES:
{json.dumps(shap_top_features)}

DATASET PROFILE:
- Numeric columns: {profile.get('numeric_cols', [])}
- Categorical columns: {profile.get('categorical_cols', [])}
- Missing values: {int(profile.get('missing', pd.Series()).sum())}
- Duplicates: {profile.get('duplicates', 0)}

EXISTING INSIGHTS:
{chr(10).join(business_insights)}

Write a concise business-level explanation (max 300 words).
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a senior business data scientist."},
            {"role": "user", "content": context}
        ],
        temperature=0.4,
        max_tokens=500
    )

    return response.choices[0].message.content


# ======================================================
# CHAT FIXED (🔥 MAIN BUG FIX HERE)

def chat_with_data(
    api_key: str,
    user_message: str,
    chat_history: list,
    model_card: dict,
    profile: dict,
    df_sample: pd.DataFrame,
    problem_type: str,
    target_col: str,
    business_insights: list
) -> str:

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # =========================
    # SAFE DEFAULTS
    # =========================
    chat_history = chat_history or []
    business_insights = business_insights or []

    cols = df_sample.columns.tolist()
    sample = df_sample.head(3).to_dict()

    insights_text = "\n".join(business_insights) if business_insights else "No insights available"

    # =========================
    # 🔥 SYSTEM PROMPT (FINAL)
    # =========================
    system_prompt = f"""
You are a senior business data analyst and strategy consultant.

STRICT RULES:
- Only use the provided dataset context
- Do NOT hallucinate or assume anything not in data
- Base answers on patterns, trends, and insights
- Speak like a business advisor (clear, concise, professional)

Always structure your answer like:

📊 Insight:
(what is happening)

📉 Explanation:
(why it is happening based on data)

💡 Action:
(what should be done)

CONTEXT:
Target variable: {target_col}
Problem type: {problem_type}

Available columns:
{cols}

Key model insights:
{insights_text}

Sample data:
{sample}
"""

    # =========================
    # USER MESSAGE
    # =========================
    user_prompt = f"Question: {user_message}"

    # =========================
    # BUILD MESSAGE FLOW
    # =========================
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    # =========================
    # CALL MODEL
    # =========================
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=400
    )

    reply = response.choices[0].message.content

    # =========================
    # SAVE HISTORY
    # =========================
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": reply})

    return reply




