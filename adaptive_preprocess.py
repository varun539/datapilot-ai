import pandas as pd


def prepare_shopify_like_data(df):
    """
    Converts ANY ecommerce dataset into a standard format:
    Date | Revenue | Orders | Quantity
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
        raise ValueError("No date column found")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # =========================
    # 2. DETECT REVENUE
    # =========================
    revenue_col = None

    for col in df.columns:
        name = col.lower()
        if "price" in name or "sales" in name or "amount" in name:
            revenue_col = col
            break

    # fallback: Quantity * UnitPrice
    if revenue_col is None:
        if "Quantity" in df.columns and "UnitPrice" in df.columns:
            df["Revenue"] = df["Quantity"] * df["UnitPrice"]
            revenue_col = "Revenue"
        else:
            raise ValueError("No revenue column found")

    # =========================
    # 3. DETECT ORDER ID
    # =========================
    order_col = None
    for col in df.columns:
        if "order" in col.lower() or "invoice" in col.lower():
            order_col = col
            break

    # =========================
    # 4. AGGREGATE
    # =========================
    agg_dict = {
        revenue_col: "sum"
    }

    if order_col:
        agg_dict[order_col] = "nunique"

    if "Quantity" in df.columns:
        agg_dict["Quantity"] = "sum"

    grouped = df.groupby(df[date_col].dt.date).agg(agg_dict).reset_index()

    # rename
    grouped.columns = ["Date", "Revenue", "Orders", "Quantity"][:len(grouped.columns)]

    # =========================
    # 5. EXTRA METRICS
    # =========================
    if "Orders" in grouped.columns:
        grouped["Avg_Order_Value"] = grouped["Revenue"] / (grouped["Orders"] + 1e-6)

    return grouped
