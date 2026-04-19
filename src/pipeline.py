import pandas as pd
import numpy as np


def prepare_features(df, profile, target_col, training=True, feature_schema=None):
    """
    REAL WORLD pipeline — zero leakage
    Works for ANY business dataset (Shopify, Walmart, retail)
    
    Business logic:
    - External factors (CPI, fuel, temperature, holidays)
    - Time patterns (seasonality, quarter, week)
    - Safe historical lags (4+ weeks back only)
    - NO store averages, NO lag_1, NO rolling mean
    """

    df = df.copy()

    # ======================================================
    # 1. CLEAN
    # ======================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # 2. DROP LEAKY + ID COLUMNS
    # Store column = essentially encodes avg sales per store
    # = leakage! Drop it.
    # ======================================================
    leaky_patterns = [
        "store", "uuid", "invoice", "record",
        "reference", "customer_id", "order_id", "row_id"
    ]
    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        cl = col.lower().replace(" ", "_")
        if any(cl == k or cl.startswith(k+"_") or cl.endswith("_"+k)
               for k in leaky_patterns):
            drop_cols.append(col)

    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ======================================================
    # 3. DATE FEATURE ENGINEERING
    # ======================================================
    date_col = None
    for col in list(df.columns):
        if col == target_col:
            continue
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(
                    df[col], dayfirst=True, errors="coerce"
                )
                if parsed.notna().mean() > 0.7:
                    date_col = col

                    df["year"]        = parsed.dt.year
                    df["month"]       = parsed.dt.month
                    df["week"]        = parsed.dt.isocalendar().week.astype(int)
                    df["quarter"]     = parsed.dt.quarter
                    df["dayofweek"]   = parsed.dt.dayofweek
                    df["is_weekend"]  = parsed.dt.dayofweek.isin([5,6]).astype(int)
                    df["is_q4"]       = (parsed.dt.quarter == 4).astype(int)
                    df["is_q1"]       = (parsed.dt.quarter == 1).astype(int)

                    # Cyclic seasonality
                    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
                    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
                    df["week_sin"]    = np.sin(2 * np.pi * df["week"]  / 52)
                    df["week_cos"]    = np.cos(2 * np.pi * df["week"]  / 52)

                    df.drop(columns=[col], inplace=True)
                    break
            except Exception:
                pass

    # ======================================================
    # 4. SORT BY TIME only (no Store since we dropped it)
    # ======================================================
    sort_cols = []
    if "year" in df.columns:
        sort_cols.append("year")
    if "week" in df.columns:
        sort_cols.append("week")
    elif "month" in df.columns:
        sort_cols.append("month")
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # ======================================================
    # 5. SAFE LAG FEATURES
    # lag_4 = 4 weeks ago, lag_8 = 8 weeks ago
    # These are what a business owner ALREADY KNOWS
    # No leakage — pure historical signal
    # ======================================================
    if target_col in df.columns and date_col:
        df["lag_4"] = df[target_col].shift(4)
        df["lag_8"] = df[target_col].shift(8)

        # Momentum: was business improving or declining?
        df["sales_momentum"] = df["lag_4"] - df["lag_8"]

        # Volatility: how stable is business?
        df["volatility"] = (
            df[target_col]
            .shift(4)
            .rolling(4, min_periods=2)
            .std()
            .fillna(0)
        )

    # ======================================================
    # 6. DROP NaN
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
    # 10. SCHEMA ALIGNMENT (prediction only)
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X, y
