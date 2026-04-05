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
# ======================================================
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

    client = get_client(api_key)

    # ✅ CRITICAL FIXES
    chat_history = chat_history or []
    model_card = model_card or {}
    business_insights = business_insights or []

    system_prompt = f"""
You are DataPilot AI.

DATASET:
- Rows: {model_card.get('rows', 'unknown')}
- Features: {model_card.get('features', 'unknown')}
- Target: {target_col}
- Problem: {problem_type}

MODEL: {model_card.get('model', 'unknown')}
PERFORMANCE: {json.dumps(model_card.get('performance', {}))}

SAMPLE:
{df_sample.head(3).to_string()}

INSIGHTS:
{chr(10).join(business_insights) if business_insights else "None"}

Answer clearly and concisely.
"""

    # ✅ FIXED MESSAGE BUILDING
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)   # SAFE NOW
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )

    reply = response.choices[0].message.content

    # ✅ SAVE CHAT HISTORY (IMPORTANT)
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": reply})

    return reply


# ======================================================
# TARGET SUGGESTION
# ======================================================
def suggest_target_column(api_key: str, columns: list, df_sample: pd.DataFrame) -> str:

    client = get_client(api_key)

    prompt = f"""
Columns: {columns}
Sample:
{df_sample.head(3).to_string()}

Which is target column? Return ONLY column name.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )

    suggested = response.choices[0].message.content.strip()

    return suggested if suggested in columns else columns[-1]


# ======================================================
# DATASET DIAGNOSIS
# ======================================================
def diagnose_dataset(
    api_key: str,
    profile: dict,
    quality_score: int,
    quality_messages: list
) -> str:

    client = get_client(api_key)

    prompt = f"""
Score: {quality_score}/100
Issues: {quality_messages}

Give 3 bullet insights.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )

    return response.choices[0].message.content
