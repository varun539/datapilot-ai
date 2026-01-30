import pandas as pd
import numpy as np


# ======================================================
# 🧠 MAIN FEATURE PIPELINE
# ======================================================
def prepare_features(
    df_raw,
    profile,
    target_col=None,
    training=True,
    feature_schema=None,
    max_corr=0.95
):
    """
    Universal feature pipeline for:
    - Training
    - Prediction
    - SHAP
    """

    # -------------------------
    # 1️⃣ COPY DATA
    # -------------------------
    df = df_raw.copy()

    # -------------------------
    # 2️⃣ DROP ID / LEAKAGE COLUMNS
    # -------------------------
    id_keywords = ["id", "uuid", "index", "code", "number"]

    drop_cols = []
    for col in df.columns:
        col_lower = col.lower()

        # Name-based ID detection
        if any(k in col_lower for k in id_keywords):
            drop_cols.append(col)
            continue

        # High-cardinality detection
        if df[col].nunique() / len(df) > 0.98:
            drop_cols.append(col)

    df.drop(columns=list(set(drop_cols)), errors="ignore", inplace=True)

    # -------------------------
    # 3️⃣ HANDLE DATETIME PROPERLY
    # -------------------------
    datetime_cols = profile.get("datetime_cols", [])

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_week"] = df[col].dt.isocalendar().week.astype("int")
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)

    # Drop raw datetime columns
    df.drop(columns=datetime_cols, errors="ignore", inplace=True)

    # -------------------------
    # 4️⃣ SEPARATE FEATURE TYPES
    # -------------------------
    numeric_cols = [
        c for c in profile["numeric_cols"]
        if c in df.columns and c != target_col
    ]

    categorical_cols = [
        c for c in profile["categorical_cols"]
        if c in df.columns
        and c != target_col
        and c not in datetime_cols
    ]

    # -------------------------
    # 5️⃣ MISSING VALUE HANDLING
    # -------------------------
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # -------------------------
    # 6️⃣ REMOVE HIGHLY CORRELATED NUMERIC FEATURES
    # -------------------------
    if training and len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        drop_corr = [
            column for column in upper.columns
            if any(upper[column] > max_corr)
        ]

        df.drop(columns=drop_corr, errors="ignore", inplace=True)
        numeric_cols = [c for c in numeric_cols if c not in drop_corr]

    # -------------------------
    # 7️⃣ BASE FEATURE FRAME
    # -------------------------
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()

    # -------------------------
    # 8️⃣ ENCODE CATEGORICALS
    # -------------------------
    X = pd.get_dummies(X, drop_first=True)

    # -------------------------
    # 9️⃣ NUMERIC SAFETY (SHAP SAFE)
    # -------------------------
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # -------------------------
    # 🔁 10️⃣ SCHEMA ALIGNMENT (PREDICTION)
    # -------------------------
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0

        X = X[feature_schema]

    return X
