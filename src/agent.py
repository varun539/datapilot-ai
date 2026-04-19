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
# 💼 EXECUTIVE SUMMARY (PREMIUM STYLE)
# ======================================================
def generate_agent_narrative(
    api_key,
    model_card,
    business_insights,
    profile,
    shap_top_features,
    problem_type,
    target_col
) -> str:

    client = get_client(api_key)

    model_card = model_card or {}
    business_insights = business_insights or []
    shap_top_features = shap_top_features or []

    context = f"""
You are a senior business consultant advising an e-commerce or retail business owner.

Write in SIMPLE English.
NO technical terms.
NO ML jargon.

BUSINESS CONTEXT:
- Target metric: {target_col}
- Dataset size: {model_card.get('rows', 'unknown')}

TOP DRIVERS:
{json.dumps(shap_top_features)}

KEY INSIGHTS:
{chr(10).join(business_insights)}

Write a sharp executive summary:

1. What is driving performance (plain English)
2. What actions should be taken immediately (2–3)
3. What risks to watch

Keep it concise (~120–150 words).
Make it sound like a paid consultant.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior business consultant."},
            {"role": "user", "content": context}
        ],
        temperature=0.4,
        max_tokens=300
    )

    return response.choices[0].message.content


# ======================================================
# 💬 CHAT WITH DATA (CONSULTANT MODE)
# ======================================================
def chat_with_data(
    api_key,
    user_message,
    chat_history,
    model_card,
    profile,
    df_sample,
    problem_type,
    target_col,
    business_insights
) -> str:

    client = get_client(api_key)

    if df_sample is None or df_sample.empty:
        return "No dataset loaded. Please upload data first."

    chat_history = chat_history or []
    business_insights = business_insights or []

    cols = df_sample.columns.tolist()
    sample = df_sample.head(3).to_string()

    insights_text = "\n".join(business_insights) if business_insights else "No insights yet"

    system_prompt = f"""
You are a senior business advisor (top consulting firm level).

STRICT RULES:
- No technical jargon
- No ML terms
- No guessing
- Only use provided data

Always respond in this format:

📊 Insight:
(what is happening)

📉 Explanation:
(why based on data)

💡 Action:
(what to do next — concrete steps)

⚠️ Risk:
(optional — what could go wrong)

BUSINESS CONTEXT:
Target: {target_col}
Problem: {problem_type}

Available columns:
{cols}

Key insights:
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
        max_tokens=350
    )

    reply = response.choices[0].message.content

    # ✅ SAVE CHAT HISTORY SAFELY
    if isinstance(chat_history, list):
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": reply})

    return reply


# ======================================================
# 🎯 SMART TARGET SELECTION
# ======================================================
def suggest_target_column(api_key, columns, df_sample):

    if not columns:
        return None

    priority = [
        "Revenue", "Sales", "Weekly_Sales",
        "Profit", "Orders", "Churn"
    ]

    for col in priority:
        if col in columns:
            return col

    # AI fallback
    try:
        client = get_client(api_key)

        prompt = f"""
Columns: {columns}

Sample:
{df_sample.head(3).to_string()}

Which column is the main business metric?
Return ONLY the column name.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )

        result = response.choices[0].message.content.strip()

        return result if result in columns else columns[-1]

    except:
        return columns[-1]


# ======================================================
# 📊 DATA QUALITY ADVISOR
# ======================================================
def diagnose_dataset(api_key, profile, quality_score, quality_messages):

    client = get_client(api_key)

    prompt = f"""
A business owner uploaded their dataset.

Quality score: {quality_score}/100
Issues: {quality_messages}

Give 3 simple, practical suggestions to improve their data.

No technical language.
Speak like a consultant helping a small business.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )

    return response.choices[0].message.content
