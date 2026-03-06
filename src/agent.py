# src/agent.py
# 🤖 DataPilot Agentic AI Layer — Powered by OpenAI GPT-4o
# Plugs directly into your existing pipeline, automl, impact, eda modules

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
# Replaces basic impact.py with GPT-4o powered insights
# ======================================================
def generate_agent_narrative(
    api_key: str,
    model_card: dict,
    business_insights: list,
    profile: dict,
    shap_top_features: list,  # [(feat, shap_value), ...]
    problem_type: str,
    target_col: str
) -> str:
    """
    Takes your existing DataPilot outputs and generates
    a senior data scientist level narrative using GPT-4o.
    """

    client = get_client(api_key)

    # Build context from YOUR existing modules
    context = f"""
You are a senior data scientist reviewing an AutoML analysis.

MODEL CARD:
- Model: {model_card.get('model')}
- Problem Type: {problem_type}
- Target: {target_col}
- Rows: {model_card.get('rows')}
- Features: {model_card.get('features')}
- Performance: {json.dumps(model_card.get('performance', {}))}

TOP SHAP FEATURES (from SHAP analysis):
{json.dumps(shap_top_features)}

DATASET PROFILE:
- Numeric columns: {profile.get('numeric_cols', [])}
- Categorical columns: {profile.get('categorical_cols', [])}
- Missing values: {int(profile.get('missing', pd.Series()).sum())}
- Duplicates: {profile.get('duplicates', 0)}

EXISTING INSIGHTS (basic):
{chr(10).join(business_insights)}

YOUR TASK:
Write a professional, executive-level analysis in 4 sections:

1. **Executive Summary** (2-3 sentences, what the model found, for a non-technical business owner)
2. **Key Drivers** (explain top 3 SHAP features in plain English, what they mean for the business)
3. **Actionable Recommendations** (3 concrete actions the business should take based on this model)
4. **Risk & Limitations** (2 honest limitations, what to watch out for)

Rules:
- No technical jargon unless explained
- No claims of causation, only association
- Be specific to the target variable: {target_col}
- Sound like a McKinsey consultant, not a textbook
- Keep it under 300 words total
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a world-class data scientist who explains ML results to business executives. You are precise, confident, and business-focused."
            },
            {
                "role": "user",
                "content": context
            }
        ],
        temperature=0.4,
        max_tokens=600
    )

    return response.choices[0].message.content


# ======================================================
# CHAT: Ask questions about YOUR data
# Full conversation memory support
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
    """
    Chat interface — user can ask anything about their data.
    Maintains conversation history for context.
    """

    client = get_client(api_key)

    # Build system context from YOUR DataPilot state
    system_prompt = f"""
You are DataPilot AI — an intelligent data science assistant embedded in an AutoML platform.

You have already analyzed the user's dataset. Here is what you know:

DATASET:
- Rows: {model_card.get('rows', 'unknown')}
- Features: {model_card.get('features', 'unknown')}
- Target variable: {target_col}
- Problem type: {problem_type}
- Numeric columns: {profile.get('numeric_cols', [])}
- Categorical columns: {profile.get('categorical_cols', [])}

BEST MODEL: {model_card.get('model', 'unknown')}
PERFORMANCE: {json.dumps(model_card.get('performance', {}))}

SAMPLE DATA (first 3 rows):
{df_sample.head(3).to_string()}

KEY INSIGHTS ALREADY FOUND:
{chr(10).join(business_insights) if business_insights else 'Not yet generated'}

YOUR BEHAVIOR:
- Answer questions about this specific dataset and model
- If asked about predictions, explain what factors drive them
- If asked "why", explain using the SHAP insights
- If asked about business decisions, give concrete advice
- Always be honest about uncertainty
- Keep answers concise (2-4 sentences unless asked for detail)
- Never make up numbers not in the context above
"""

    # Build messages with history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=400
    )

    return response.choices[0].message.content


# ======================================================
# SMART COLUMN SUGGESTER
# Uses GPT-4o to suggest best target column
# ======================================================
def suggest_target_column(
    api_key: str,
    columns: list,
    df_sample: pd.DataFrame
) -> str:
    """
    Looks at column names + sample data and suggests
    the most likely target variable for ML.
    """

    client = get_client(api_key)

    prompt = f"""
A user uploaded a dataset with these columns: {columns}

Sample data (3 rows):
{df_sample.head(3).to_string()}

Which column is most likely the TARGET variable for machine learning prediction?
Respond with ONLY the column name, nothing else.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )

    suggested = response.choices[0].message.content.strip()

    # Safety: make sure it's a real column
    if suggested in columns:
        return suggested
    return columns[-1]  # fallback to last column


# ======================================================
# DATASET HEALTH DOCTOR
# Deep analysis beyond your data_quality.py score
# ======================================================
def diagnose_dataset(
    api_key: str,
    profile: dict,
    quality_score: int,
    quality_messages: list
) -> str:
    """
    GPT-4o powered dataset health report.
    Goes beyond the numeric score in data_quality.py.
    """

    client = get_client(api_key)

    prompt = f"""
You are reviewing a dataset before machine learning training.

QUALITY SCORE: {quality_score}/100
ISSUES FOUND: {quality_messages}
ROWS: {profile.get('rows')}
COLUMNS: {profile.get('columns')}
NUMERIC FEATURES: {profile.get('numeric_cols', [])}
CATEGORICAL FEATURES: {profile.get('categorical_cols', [])}
MISSING VALUES: {int(profile.get('missing', pd.Series()).sum())}
DUPLICATES: {profile.get('duplicates', 0)}

Give a 3-bullet diagnosis:
• What is healthy about this dataset
• What needs attention before modeling
• One specific recommendation to improve model performance

Keep each bullet to 1 sentence. Be direct.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )

    return response.choices[0].message.content
