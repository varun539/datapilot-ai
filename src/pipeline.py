import pandas as pd
import numpy as np


def prepare_features(df, profile, target_col, training=True, feature_schema=None):
    """
    Smart feature pipeline — returns (X, y) tuple
    Handles dates, lags, store features, encoding
    """

    df = df.copy()

    # ======================================================
    # 1. CLEAN — Remove inf values
    # ======================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # 2. DROP USELESS COLUMNS
    # ======================================================
    id_keywords = [
        "uuid", "invoice", "record", "reference"
    ]
    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        cl = col.lower().replace(" ", "_")
        if any(k in cl for k in id_keywords):
            drop_cols.append(col)
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
                # Try multiple date formats
                parsed = pd.to_datetime(
                    df[col],
                    dayfirst=True,      # handles DD-MM-YYYY like 26-10-2012
                    errors="coerce"
                )
                if parsed.notna().mean() > 0.7:
                    date_col = col
                    df[col] = parsed
                    df[f"{col}_year"]       = parsed.dt.year
                    df[f"{col}_month"]      = parsed.dt.month
                    df[f"{col}_week"]       = parsed.dt.isocalendar().week.astype(int)
                    df[f"{col}_day"]        = parsed.dt.day
                    df[f"{col}_dayofweek"]  = parsed.dt.dayofweek
                    df[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
                    df[f"{col}_quarter"]    = parsed.dt.quarter
                    df.drop(columns=[col], inplace=True)
                    break
            except Exception:
                pass

    # ======================================================
    # 4. SORT BY STORE + DATE (time series integrity)
    # ======================================================
    sort_cols = []
    if "Store" in df.columns:
        sort_cols.append("Store")
    date_features = [c for c in df.columns if c.endswith("_year") or c.endswith("_month")]
    if date_features:
        sort_cols.extend(date_features[:2])
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # ======================================================
    # 5. STORE FEATURES (huge R² boost for retail data)
    # ======================================================
    if "Store" in df.columns and target_col in df.columns:
        df["store_avg"]    = df.groupby("Store")[target_col].transform("mean")
        df["store_std"]    = df.groupby("Store")[target_col].transform("std").fillna(0)
        df["store_median"] = df.groupby("Store")[target_col].transform("median")

    # ======================================================
    # 6. LAG FEATURES (time intelligence)
    # ======================================================
    if "Store" in df.columns and target_col in df.columns and date_col is not None:
        df["lag_1"] = df.groupby("Store")[target_col].shift(1)
        df["lag_2"] = df.groupby("Store")[target_col].shift(2)
        df["lag_4"] = df.groupby("Store")[target_col].shift(4)

        # Rolling features
        df["rolling_mean_4"] = (
            df.groupby("Store")[target_col]
            .shift(1)
            .rolling(4, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["rolling_std_4"] = (
            df.groupby("Store")[target_col]
            .shift(1)
            .rolling(4, min_periods=1)
            .std()
            .fillna(0)
            .reset_index(level=0, drop=True)
        )

    # ======================================================
    # 7. DROP ROWS WITH NaN FROM LAGS
    # ======================================================
    df = df.dropna(subset=[target_col])
    df = df.reset_index(drop=True)

    # ======================================================
    # 8. SPLIT X and y
    # ======================================================
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    X = df.drop(columns=[target_col])

    # ======================================================
    # 9. HANDLE CATEGORICAL COLUMNS
    # ======================================================
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        n_unique = X[col].nunique()
        if n_unique <= 20:
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
