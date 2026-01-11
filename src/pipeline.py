import pandas as pd
import numpy as np
from src.feature_engineering import engineer_features


def prepare_features(
    df_raw,
    profile,
    target_col=None,
    training=True,
    feature_schema=None,
    max_features=40
):
    """
    Universal feature pipeline used for:
    - Training
    - Prediction
    - SHAP
    """

    # ================================
    # 🚮 FORCE REMOVE ID / USELESS COLUMNS
    # ================================
    df = df_raw.copy()

    id_keywords = ["id", "uuid", "index", "code", "number"]
    drop_cols = []

    for col in df.columns:
        col_lower = col.lower()

        # Rule 1: name-based kill
        if any(k in col_lower for k in id_keywords):
            drop_cols.append(col)
            continue

        # Rule 2: almost-unique kill
        if df[col].nunique() / len(df) > 0.98:
            drop_cols.append(col)

    drop_cols = list(set(drop_cols))

    if drop_cols:
        print("🔥 Dropping ID / useless columns:", drop_cols)
        df = df.drop(columns=drop_cols)

    # ================================
    # ✅ AUTO MISSING HANDLING
    # ================================
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # ================================
    # 🧠 RE-DETECT COLUMN TYPES (CRITICAL FIX)
    # ================================
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove target from features
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    # ================================
    # BASE FEATURES
    # ================================
    X_base = df[numeric_cols + categorical_cols].copy()

    # ================================
    # FEATURE ENGINEERING
    # ================================
    X_fe = engineer_features(
        df=df,
        numeric_cols=numeric_cols,
        date_cols=profile.get("datetime_cols", []),
        target_col=target_col,
        max_features=max_features
    )

    X = pd.concat([X_base, X_fe], axis=1)

    # ================================
    # ENCODING
    # ================================
    X = pd.get_dummies(X, drop_first=True)

    # ================================
    # NUMERIC SAFETY
    # ================================
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ================================
    # SCHEMA ALIGNMENT (PREDICTION MODE)
    # ================================
    if not training and feature_schema is not None:

        # Add missing columns
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0

        # Keep only training columns
        X = X[feature_schema]

    return X
