import pandas as pd
import numpy as np


# ======================================================
# 🛒 MAIN FUNCTION — UNIVERSAL PREPROCESS
# ======================================================
def prepare_shopify_like_data(df):
    """
    Converts ANY ecommerce dataset into structured format:
    Date | Revenue | Orders | Quantity | Avg_Order_Value
    """

    df = df.copy()

    # =========================
    # 1. DETECT DATE COLUMN
    # =========================
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        raise ValueError("❌ No date column found")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # =========================
    # 2. DETECT REVENUE
    # =========================
    revenue_col = None

    for col in df.columns:
        name = col.lower()
        if any(k in name for k in ["revenue", "sales", "amount", "total_price"]):
            revenue_col = col
            break

    # fallback → Quantity * UnitPrice
    if revenue_col is None:
        if "Quantity" in df.columns and "UnitPrice" in df.columns:
            df["Revenue"] = df["Quantity"] * df["UnitPrice"]
            revenue_col = "Revenue"
        else:
            raise ValueError("❌ No revenue column found")

    # =========================
    # 3. DETECT ORDER COLUMN
    # =========================
    order_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["order", "invoice"]):
            order_col = col
            break

    # =========================
    # 4. AGGREGATE TO DAILY LEVEL
    # =========================
    agg_dict = {
        revenue_col: "sum"
    }

    if order_col:
        agg_dict[order_col] = "nunique"

    if "Quantity" in df.columns:
        agg_dict["Quantity"] = "sum"

    grouped = df.groupby(df[date_col].dt.date).agg(agg_dict).reset_index()

    # Rename columns safely
    new_cols = ["Date", "Revenue"]
    if order_col:
        new_cols.append("Orders")
    if "Quantity" in agg_dict:
        new_cols.append("Quantity")

    grouped.columns = new_cols

    # =========================
    # 5. DERIVED METRICS
    # =========================
    if "Orders" in grouped.columns:
        grouped["Avg_Order_Value"] = grouped["Revenue"] / (grouped["Orders"] + 1e-6)

    return grouped


# ======================================================
# 👤 CHURN CREATION
# ======================================================
def create_churn_features(df, inactivity_days=30):
    """
    Creates churn dataset at CUSTOMER LEVEL

    Output:
    CustomerID | Recency | Frequency | Monetary | Churn
    """

    if "CustomerID" not in df.columns:
        return None

    # detect date column
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # revenue calculation if missing
    if "Revenue" not in df.columns:
        if "Quantity" in df.columns and "UnitPrice" in df.columns:
            df["Revenue"] = df["Quantity"] * df["UnitPrice"]
        else:
            df["Revenue"] = 0

    last_date = df[date_col].max()

    # =========================
    # RFM FEATURES
    # =========================
    rfm = df.groupby("CustomerID").agg({
        date_col: lambda x: (last_date - x.max()).days,
        "CustomerID": "count",
        "Revenue": "sum"
    }).rename(columns={
        date_col: "Recency",
        "CustomerID": "Frequency",
        "Revenue": "Monetary"
    }).reset_index()

    # =========================
    # CHURN LABEL
    # =========================
    rfm["Churn"] = (rfm["Recency"] > inactivity_days).astype(int)

    return rfm


# ======================================================
# 🎯 MASTER FUNCTION (USE THIS IN APP)
# ======================================================
def adaptive_preprocess(df, mode="revenue"):
    """
    mode:
    - "revenue" → returns time-series dataset
    - "churn"   → returns customer-level dataset
    """

    if mode == "revenue":
        return prepare_shopify_like_data(df)

    elif mode == "churn":
        churn_df = create_churn_features(df)
        if churn_df is None:
            raise ValueError("❌ Cannot create churn — missing CustomerID or Date")
        return churn_df

    else:
        raise ValueError("Invalid mode")
