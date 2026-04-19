import pandas as pd
import numpy as np


def prepare_features(df, profile, target_col, training=True, feature_schema=None):
    """
    REAL WORLD pipeline — zero leakage
    Designed for Shopify/retail business owners
    
    Features that make BUSINESS SENSE:
    - When did the sale happen? (date features)
    - Is it a holiday? (holiday flag)  
    - What are external conditions? (temp, fuel, CPI, unemployment)
    - What was the trend before? (lag_4, lag_8 — safe lags)
    
    NO store_avg, NO store_median (leakage!)
    NO lag_1, NO rolling_mean (too correlated!)
    """

    df = df.copy()

    # ======================================================
    # 1. CLEAN
    # ======================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # 2. DROP PURE ID COLUMNS
    # ======================================================
    drop_always = ["uuid", "invoice", "record", "reference",
                   "customer_id", "order_id", "row_id"]
    drop_cols = [
        c for c in df.columns
        if c != target_col and
        any(k == c.lower().replace(" ","_") for k in drop_always)
    ]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ======================================================
    # 3. DATE FEATURE ENGINEERING
    # Extracts business-meaningful time features
    # ======================================================
    date_col = None
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(
                    df[col], dayfirst=True, errors="coerce"
                )
                if parsed.notna().mean() > 0.7:
                    date_col = col

                    # Raw time features
                    df["year"]       = parsed.dt.year
                    df["month"]      = parsed.dt.month
                    df["week"]       = parsed.dt.isocalendar().week.astype(int)
                    df["quarter"]    = parsed.dt.quarter
                    df["dayofweek"]  = parsed.dt.dayofweek
                    df["is_weekend"] = parsed.dt.dayofweek.isin([5,6]).astype(int)

                    # Cyclic encoding — captures seasonality properly
                    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
                    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
                    df["week_sin"]   = np.sin(2 * np.pi * df["week"]  / 52)
                    df["week_cos"]   = np.cos(2 * np.pi * df["week"]  / 52)

                    # Q4 flag — holiday shopping season
                    df["is_q4"]      = (df["quarter"] == 4).astype(int)

                    df.drop(columns=[col], inplace=True)
                    break
            except Exception:
                pass

    # ======================================================
    # 4. SORT BY STORE + TIME (critical for lag features)
    # ======================================================
    sort_cols = []
    if "Store" in df.columns:
        sort_cols.append("Store")
    if "year" in df.columns and "week" in df.columns:
        sort_cols += ["year", "week"]
    elif "year" in df.columns and "month" in df.columns:
        sort_cols += ["year", "month"]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # ======================================================
    # 5. SAFE LAG FEATURES — NO LEAKAGE
    # Only use lag_4 (1 month ago) and lag_8 (2 months ago)
    # These represent what a business owner ALREADY KNOWS
    # when making decisions
    # ======================================================
    if "Store" in df.columns and target_col in df.columns and date_col:
        grp = df.groupby("Store")[target_col]

        # What were sales 1 month ago?
        df["lag_4"]  = grp.shift(4)

        # What were sales 2 months ago?
        df["lag_8"]  = grp.shift(8)

        # Was last month better or worse than 2 months ago?
        df["sales_momentum"] = df["lag_4"] - df["lag_8"]

        # Rolling std — how volatile is this store?
        # (shifted by 4 to avoid leakage)
        df["volatility_4w"] = (
            grp.shift(4)
            .rolling(4, min_periods=2)
            .std()
            .fillna(0)
            .reset_index(level=0, drop=True)
        )

    # ======================================================
    # 6. DROP NaN ROWS FROM LAGS
    # ======================================================
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # ======================================================
    # 7. SPLIT X, y
    # ======================================================
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    X = df.drop(columns=[target_col])

    # ======================================================
    # 8. CATEGORICAL ENCODING
    # ======================================================
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if X[col].nunique() <= 20:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X.drop(columns=[col], inplace=True)

    # ======================================================
    # 9. FINAL CLEAN
    # ======================================================
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # ======================================================
    # 10. SCHEMA ALIGNMENT (prediction mode)
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X, y
