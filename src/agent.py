from openai import OpenAI
import pandas as pd


# ======================================================
# INIT CLIENT
# ======================================================
def get_client(api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")
    return OpenAI(api_key=api_key)


# ======================================================
# 💬 CHAT WITH DATA (CLEAN VERSION)
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

    # Safety checks
    if not api_key:
        return "⚠️ API key missing"

    if df_sample is None or df_sample.empty:
        return "No dataset loaded. Please upload data first."

    client = get_client(api_key)

    chat_history = chat_history or []
    business_insights = business_insights or []

    # Context preparation
    cols = df_sample.columns.tolist()
    sample = df_sample.head(3).to_string()

    insights_text = "\n".join(business_insights) if business_insights else "No insights available"

    # ======================================================
    # SYSTEM PROMPT (SIMPLIFIED FOR RECRUITER VERSION)
    # ======================================================
    system_prompt = f"""
You are a data analyst helping interpret machine learning results.

Explain clearly and concisely.

Context:
- Target: {target_col}
- Problem type: {problem_type}

Columns:
{cols}

Insights:
{insights_text}

Sample data:
{sample}

Rules:
- No jargon
- No assumptions beyond data
- Keep answers practical and concise
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    # ======================================================
    # API CALL
    # ======================================================
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception:
        return "⚠️ AI response unavailable. Please try again."


# ======================================================
# 🎯 SMART TARGET SELECTION
# ======================================================
def suggest_target_column(api_key, columns, df_sample):

    if not columns:
        return None

    # Priority columns (business-relevant)
    priority = [
        "Revenue", "Sales", "Weekly_Sales",
        "Profit", "Orders", "Churn"
    ]

    for col in priority:
        if col in columns:
            return col

    # Fallback: choose last numeric column
    numeric_cols = df_sample.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        return numeric_cols[-1]

    return columns[-1]
