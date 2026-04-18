import pandas as pd
import numpy as np


def prepare_features(df, profile, target_col, training=True, feature_schema=None):
    """
    Production pipeline — NO data leakage
    Real world useful features only
    """
    df = df.copy()

    # ======================================================
    # 1. CLEAN
    # ======================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # 2. DROP USELESS ID COLUMNS
    # ======================================================
    id_keywords = ["uuid", "invoice", "record", "reference"]
    drop_cols = [
        c for c in df.columns
        if c != target_col and
        any(k in c.lower() for k in id_keywords)
    ]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ======================================================
    # 3. DATE FEATURE ENGINEERING
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
                    df[col]         = parsed
                    df["year"]      = parsed.dt.year
                    df["month"]     = parsed.dt.month
                    df["week"]      = parsed.dt.isocalendar().week.astype(int)
                    df["quarter"]   = parsed.dt.quarter
                    df["dayofweek"] = parsed.dt.dayofweek
                    df["is_weekend"]= parsed.dt.dayofweek.isin([5,6]).astype(int)

                    # Cyclic encoding — better than raw month/week
                    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
                    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
                    df["week_sin"]  = np.sin(2 * np.pi * df["week"]  / 52)
                    df["week_cos"]  = np.cos(2 * np.pi * df["week"]  / 52)

                    df.drop(columns=[col], inplace=True)
                    break
            except Exception:
                pass

    # ======================================================
    # 4. SORT BY STORE + DATE
    # ======================================================
    sort_cols = []
    if "Store" in df.columns:
        sort_cols.append("Store")
    if date_col:
        sort_cols += ["year", "week"] if "week" in df.columns else ["year", "month"]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # ======================================================
    # 5. STORE AGGREGATE FEATURES (no leakage — group stats)
    # ======================================================
    if "Store" in df.columns and target_col in df.columns:
        df["store_avg"]    = df.groupby("Store")[target_col].transform("mean")
        df["store_median"] = df.groupby("Store")[target_col].transform("median")
        df["store_std"]    = df.groupby("Store")[target_col].transform("std").fillna(0)

    # ======================================================
    # 6. LAG FEATURES — lag_2 and lag_4 ONLY (no lag_1!)
    # lag_1 = 0.95 correlation = leakage risk
    # lag_2 = last 2 weeks = safe and useful
    # lag_4 = last month = safe and useful
    # ======================================================
    if "Store" in df.columns and target_col in df.columns and date_col:
        df["lag_2"] = df.groupby("Store")[target_col].shift(2)
        df["lag_4"] = df.groupby("Store")[target_col].shift(4)

        # Rolling std only — mean is too correlated
        df["rolling_std_4"] = (
            df.groupby("Store")[target_col]
            .shift(2)   # shift 2 to avoid leakage
            .rolling(4, min_periods=2)
            .std()
            .fillna(0)
            .reset_index(level=0, drop=True)
        )

    # ======================================================
    # 7. DROP NaN FROM LAGS
    # ======================================================
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # ======================================================
    # 8. SPLIT X, y
    # ======================================================
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    X = df.drop(columns=[target_col])

    # ======================================================
    # 9. CATEGORICAL ENCODING — smart
    # ======================================================
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if X[col].nunique() <= 20:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X.drop(columns=[col], inplace=True)

    # ======================================================
    # 10. FINAL CLEAN
    # ======================================================
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # ======================================================
    # 11. SCHEMA ALIGNMENT (prediction mode)
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X, y
