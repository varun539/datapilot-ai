from openai import OpenAI
import json
import pandas as pd


# ======================================================
# INIT CLIENT
# ======================================================
def get_client(api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")
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

    model_card = model_card or {}
    business_insights = business_insights or []
    shap_top_features = shap_top_features or []
    profile = profile or {}

    context = f"""
You are a senior business data scientist reviewing an AutoML analysis.

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
- Missing values: {int(profile.get('missing', pd.Series()).sum()) if 'missing' in profile else 0}
- Duplicates: {profile.get('duplicates', 0)}

EXISTING INSIGHTS:
{chr(10).join(business_insights)}

Write a concise business-level explanation (max 200 words).
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior business data scientist."},
            {"role": "user", "content": context}
        ],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content


# ======================================================
# CHAT (CONSULTANT STYLE)
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

    chat_history = chat_history or []
    business_insights = business_insights or []

    if df_sample is None or df_sample.empty:
        return "No dataset available. Please upload and analyze data first."

    cols = df_sample.columns.tolist()
    sample = df_sample.head(3).to_dict()

    insights_text = "\n".join(business_insights) if business_insights else "No insights available"

    system_prompt = f"""
You are a senior business data analyst and strategy consultant.

STRICT RULES:
- Only use the provided dataset context
- Do NOT hallucinate
- Be concise and professional

Always respond in this structure:

📊 Insight:
(what is happening)

📉 Explanation:
(why based on data)

💡 Action:
(what should be done)

CONTEXT:
Target: {target_col}
Problem: {problem_type}

Columns:
{cols}

Model insights:
{insights_text}

Sample data:
{sample}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )

    reply = response.choices[0].message.content

    # save history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": reply})

    return reply


# ======================================================
# TARGET SUGGESTION (FIXED 🔥)
# ======================================================
def suggest_target_column(api_key: str, columns: list, df_sample: pd.DataFrame) -> str:

    client = get_client(api_key)

    if not columns:
        return None

    prompt = f"""
Columns: {columns}

Sample:
{df_sample.head(3).to_string()}

Return ONLY the most likely target column.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=30
        )

        result = response.choices[0].message.content.strip()

        return result if result in columns else columns[-1]

    except:
        # fallback
        return columns[-1]


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

Give 3 short actionable insights.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=120
    )

    return response.choices[0].message.content
