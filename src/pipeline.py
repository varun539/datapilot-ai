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
    Clean, safe feature pipeline:
    - Drops ID-like columns (including Store 🔥)
    - Handles datetime
    - Encodes categorical safely
    - Prevents leakage
    - Removes high correlation
    """

    # ======================================================
    # 1 COPY DATA
    # ======================================================
    df = df_raw.copy()

    # ======================================================
    # 2 DROP ID / USELESS COLUMNS (🔥 FIXED)
    # ======================================================
    id_keywords = [
        "id", "uuid", "index", "code", "number",
        "name", "email", "phone", "address",
        "zip", "postal", "order", "invoice",
        "row", "record", "key", "ref", "reference",
        "store"   # 🔥 IMPORTANT FIX
    ]

    drop_cols = []

    for col in df.columns:
        if col == target_col:
            continue

        col_lower = col.lower().replace(" ", "_").replace("-", "_")

        # Keyword-based drop
        if any(k in col_lower for k in id_keywords):
            drop_cols.append(col)
            continue

        # Drop constant columns
        if df[col].nunique() <= 1:
            drop_cols.append(col)
            continue

        # High-cardinality text
        if df[col].dtype == "object":
            if df[col].nunique() > 50:
                drop_cols.append(col)
                continue

        # Near-unique numeric (ID-like)
        if training and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() / len(df) > 0.98:
                drop_cols.append(col)

    drop_cols = [c for c in drop_cols if c != target_col]
    df.drop(columns=list(set(drop_cols)), errors="ignore", inplace=True)

    # ======================================================
    # 3 DATETIME FEATURES
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
        df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)

    df.drop(columns=datetime_cols, errors="ignore", inplace=True)

    # ======================================================
    # 4 FEATURE TYPES
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
    # 5 MISSING VALUES
    # ======================================================
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # ======================================================
    # 6 REMOVE HIGH CORRELATION
    # ======================================================
    if training and len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        drop_corr = [
            col for col in upper.columns
            if any(upper[col] > max_corr)
        ]

        df.drop(columns=drop_corr, errors="ignore", inplace=True)
        numeric_cols = [c for c in numeric_cols if c not in drop_corr]

    # ======================================================
    # 7 BUILD FEATURE FRAME (🔥 FIXED TARGET SAFETY)
    # ======================================================
    feature_cols = numeric_cols + categorical_cols
    feature_cols = [c for c in feature_cols if c != target_col]  # 🔥 FIX

    X = df[feature_cols].copy()

    if X.shape[1] == 0:
        if not training and feature_schema is not None:
            return pd.DataFrame(0, index=[0], columns=feature_schema)
        return pd.DataFrame(index=df.index)

    # ======================================================
    # 8 ONE-HOT ENCODING
    # ======================================================
    if categorical_cols:
        safe_categorical = []
        for col in categorical_cols:
            if col not in X.columns:
                continue

            if X[col].nunique() <= 20:
                safe_categorical.append(col)
            else:
                X.drop(columns=[col], errors="ignore", inplace=True)

        if safe_categorical:
            X = pd.get_dummies(X, columns=safe_categorical, drop_first=True)

    # ======================================================
    # 9 NUMERIC SAFETY
    # ======================================================
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ======================================================
    # 10 SCHEMA ALIGNMENT
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X
