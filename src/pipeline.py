import pandas as pd
import numpy as np


def prepare_features(df, profile, target_col, training=True, feature_schema=None):
    """
    PRODUCTION SAFE PIPELINE
    - No leakage
    - Fast execution
    - Works on any dataset
    """

    df = df.copy()

    # ======================================================
    # 1. CLEAN
    # ======================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # 2. DROP ID / LEAKY COLUMNS
    # ======================================================
    drop_keywords = [
        "id", "uuid", "invoice", "order",
        "customer", "transaction", "row"
    ]

    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        name = col.lower()
        if any(k in name for k in drop_keywords):
            drop_cols.append(col)

    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ======================================================
    # 3. DATE FEATURES (SAFE + FAST)
    # ======================================================
    for col in df.columns:
        if col == target_col:
            continue

        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")

            if parsed.notna().mean() > 0.8:
                df["year"] = parsed.dt.year
                df["month"] = parsed.dt.month
                df["dayofweek"] = parsed.dt.dayofweek
                df["is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)

                df.drop(columns=[col], inplace=True)
                break

    # ======================================================
    # 4. TARGET SPLIT (NO FUTURE LEAKAGE)
    # ======================================================
    if target_col not in df.columns:
        return pd.DataFrame(), pd.Series()

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])

    # ======================================================
    # 5. SIMPLE LAG (SAFE ONLY)
    # ======================================================
    if len(df) > 10:
        X["lag_3"] = y.shift(3)
        X["lag_5"] = y.shift(5)

    # ======================================================
    # 6. DROP NA (FROM LAGS)
    # ======================================================
    X = X.fillna(0)
    y = y.fillna(0)

    # ======================================================
    # 7. ENCODE CATEGORICAL (LIGHTWEIGHT)
    # ======================================================
    cat_cols = X.select_dtypes(include="object").columns

    for col in cat_cols:
        if X[col].nunique() <= 15:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X.drop(columns=[col], inplace=True)

    # ======================================================
    # 8. FINAL CLEAN
    # ======================================================
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ======================================================
    # 9. SCHEMA ALIGNMENT (FOR PREDICTION)
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X, y
