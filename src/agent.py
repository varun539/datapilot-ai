from openai import OpenAI
import json
import pandas as pd


def get_client(api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")
    return OpenAI(api_key=api_key)


# ======================================================
# GENERATE EXECUTIVE NARRATIVE
# ======================================================
def generate_agent_narrative(
    api_key, model_card, business_insights,
    profile, shap_top_features, problem_type, target_col
) -> str:

    client = get_client(api_key)

    context = f"""
You are a senior business analyst advising a Shopify store owner or retail business.
Write in plain English — NO jargon, NO technical ML terms.

MODEL RESULTS:
- Best Model: {(model_card or {}).get('model')}
- Target: {target_col}
- R² Score: {(model_card or {}).get('performance', {}).get('R² (CV)', 'N/A')}
- Rows analyzed: {(model_card or {}).get('rows')}

TOP FACTORS DRIVING {target_col}:
{json.dumps(shap_top_features or [])}

INSIGHTS FOUND:
{chr(10).join(business_insights or [])}

Write a 150-word executive summary that:
1. States what's driving sales in plain English
2. Gives 2 concrete actions the business owner can take TODAY
3. Mentions what to watch out for

Use simple language. No "R²", no "SHAP values", no ML terms.
Talk like a business consultant, not a data scientist.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a business consultant advising retail store owners. Speak plainly and practically."},
            {"role": "user", "content": context}
        ],
        temperature=0.4,
        max_tokens=350
    )
    return response.choices[0].message.content


# ======================================================
# CHAT WITH DATA — Shopify/retail focused
# ======================================================
def chat_with_data(
    api_key, user_message, chat_history,
    model_card, profile, df_sample,
    problem_type, target_col, business_insights
) -> str:

    client = get_client(api_key)

    if df_sample is None or df_sample.empty:
        return "No dataset loaded. Please upload your data first."

    insights_text = "\n".join(business_insights or []) or "No insights yet"
    cols   = df_sample.columns.tolist()
    sample = df_sample.head(3).to_dict()

    system_prompt = f"""
You are a business data analyst helping a retail or Shopify store owner understand their sales data.

YOUR RULES:
- Speak in plain English — no ML jargon
- Be specific and actionable
- Reference actual numbers from the data when possible
- Never say "SHAP values", "R²", "model", "algorithm"
- Think like a business consultant, not a data scientist

ALWAYS respond in this format:

📊 What's happening:
(1-2 sentences — what the data shows)

📉 Why:
(1-2 sentences — the business reason)

💡 What to do:
(2-3 concrete actions)

DATA CONTEXT:
- Business target: {target_col}
- Columns available: {cols}
- Key findings: {insights_text}
- Sample data: {sample}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history or [])
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=350
    )

    reply = response.choices[0].message.content

    if isinstance(chat_history, list):
        chat_history.append({"role": "user",      "content": user_message})
        chat_history.append({"role": "assistant", "content": reply})

    return reply


# ======================================================
# SMART TARGET SUGGESTION
# ======================================================
def suggest_target_column(api_key, columns, df_sample) -> str:

    if not columns:
        return None

    # Priority list — no AI needed
    priority = [
        "Weekly_Sales", "Sales", "Revenue", "Profit",
        "Orders", "Conversions", "Target", "Price", "Churn"
    ]
    for col in priority:
        if col in columns:
            return col

    # AI fallback
    try:
        client = get_client(api_key)
        prompt = f"""
These are columns from a business dataset: {columns}

Sample data:
{df_sample.head(3).to_string()}

Which column is the most likely business KPI to predict?
Return ONLY the column name, nothing else.
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
# DATASET DIAGNOSIS
# ======================================================
def diagnose_dataset(api_key, profile, quality_score, quality_messages) -> str:

    client = get_client(api_key)

    prompt = f"""
A business owner uploaded their sales data for analysis.
Data quality score: {quality_score}/100
Issues found: {quality_messages}

Give 3 short, plain-English tips to improve their data quality.
No technical jargon. Talk like you're helping a small business owner.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content
