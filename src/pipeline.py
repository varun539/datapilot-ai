import pandas as pd
import numpy as np


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

    # ======================================================
    # 1️⃣ COPY DATA
    # ======================================================
    df = df_raw.copy()

    # ======================================================
    # 2️⃣ DROP ID / LEAKAGE COLUMNS
    # ======================================================
    id_keywords = ["id", "uuid", "index", "code", "number"]
    drop_cols = []

    for col in df.columns:
        col_lower = col.lower()

        # Name-based ID detection
        if any(k in col_lower for k in id_keywords):
            drop_cols.append(col)
            continue

        # High-cardinality leakage
        if df[col].nunique(dropna=False) / len(df) > 0.98:
            drop_cols.append(col)

    df.drop(columns=list(set(drop_cols)), errors="ignore", inplace=True)

    # ======================================================
    # 3️⃣ DATETIME FEATURE ENGINEERING (SAFE)
    # ======================================================
    datetime_cols = profile.get("datetime_cols", [])

    for col in datetime_cols:
        if col not in df.columns:
            continue

        df[col] = pd.to_datetime(df[col], errors="coerce")

        df[f"{col}_year"] = df[col].dt.year.astype("Int64")
        df[f"{col}_month"] = df[col].dt.month.astype("Int64")
        df[f"{col}_week"] = df[col].dt.isocalendar().week.astype("Int64")
        df[f"{col}_day"] = df[col].dt.day.astype("Int64")
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek.astype("Int64")

        df[f"{col}_is_weekend"] = (
            df[col].dt.dayofweek.isin([5, 6]).astype(int)
        )

    # Drop raw datetime columns (VERY IMPORTANT)
    df.drop(columns=datetime_cols, errors="ignore", inplace=True)

    # ======================================================
    # 4️⃣ FEATURE TYPE SEPARATION
    # ======================================================
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c != target_col
    ]

    categorical_cols = [
        c for c in df.columns
        if df[c].dtype == "object"
        and c != target_col
    ]

    # ======================================================
    # 5️⃣ MISSING VALUE HANDLING
    # ======================================================
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # ======================================================
    # 6️⃣ REMOVE HIGH CORRELATED NUMERIC FEATURES (TRAIN ONLY)
    # ======================================================
    if training and len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        drop_corr = [
            col for col in upper.columns
            if any(upper[col] > max_corr)
        ]

        df.drop(columns=drop_corr, errors="ignore", inplace=True)
        numeric_cols = [c for c in numeric_cols if c not in drop_corr]

    # ======================================================
    # 7️⃣ BASE FEATURE FRAME
    # ======================================================
    X = df[numeric_cols + categorical_cols].copy()

    # ======================================================
    # 8️⃣ ONE-HOT ENCODE CATEGORICALS
    # ======================================================
    X = pd.get_dummies(X, drop_first=True)

    # ======================================================
    # 9️⃣ NUMERIC SAFETY (SHAP + MODEL SAFE)
    # ======================================================
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ======================================================
    # 🔁 10️⃣ SCHEMA ALIGNMENT (PREDICTION)
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X
