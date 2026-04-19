import pandas as pd
import numpy as np


def prepare_features(df, profile, target_col, training=True, feature_schema=None):
    """
    🐐 BUSINESS INSIGHTS PIPELINE (SHOPIFY READY)

    - No leakage
    - No lag features
    - No store memorization
    - Focus on real business drivers
    """

    df = df.copy()

    # ======================================================
    # 1. CLEAN
    # ======================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # 2. DROP ID / NON-USEFUL / DOMINANT COLUMNS
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

        # 🔥 REMOVE STORE (CRITICAL FIX)
        if col.lower() == "store":
            drop_cols.append(col)

    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ======================================================
    # 3. DATE FEATURES (VERY IMPORTANT FOR SHOPIFY)
    # ======================================================
    for col in df.columns:
        if col == target_col:
            continue

        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")

            if parsed.notna().mean() > 0.8:

                df["year"] = parsed.dt.year
                df["month"] = parsed.dt.month
                df["quarter"] = parsed.dt.quarter
                df["dayofweek"] = parsed.dt.dayofweek
                df["is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)

                # 🔥 SEASONAL SIGNAL (important for sales)
                df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
                df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

                df.drop(columns=[col], inplace=True)
                break

    # ======================================================
    # 4. SPLIT TARGET
    # ======================================================
    if target_col not in df.columns:
        return pd.DataFrame(), pd.Series()

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    X = df.drop(columns=[target_col])

    # ======================================================
    # 5. NUMERIC INTERACTIONS (LIMITED + FAST)
    # ======================================================
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = numeric_cols[:6]  # keep fast

    for i in range(len(numeric_cols)):
        for j in range(i + 1, min(i + 3, len(numeric_cols))):
            c1 = numeric_cols[i]
            c2 = numeric_cols[j]
            X[f"{c1}_x_{c2}"] = X[c1] * X[c2]

    # ======================================================
    # 6. LOG TRANSFORM (STABILIZE SALES DATA)
    # ======================================================
    for col in numeric_cols:
        if (X[col] > 0).all():
            X[f"log_{col}"] = np.log1p(X[col])

    # ======================================================
    # 7. CATEGORICAL ENCODING (SAFE)
    # ======================================================
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    for col in cat_cols:
        if X[col].nunique() <= 20:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X.drop(columns=[col], inplace=True)

    # ======================================================
    # 8. FINAL CLEAN
    # ======================================================
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # ======================================================
    # 9. SCHEMA ALIGNMENT (FOR PRODUCTION)
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X, y
