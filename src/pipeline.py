import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def prepare_features(df, profile, target_col, training=True, feature_schema=None):
    """
    🔥 FINAL PIPELINE — NO LEAKAGE, REAL BUSINESS LOGIC

    ✔ Removes target-derived features
    ✔ Removes useless IDs & constants
    ✔ Keeps meaningful time patterns
    ✔ Safe lag features (ONLY past info)
    ✔ Recruiter-level clean pipeline
    """

    df = df.copy()

    # ======================================================
    # 1. CLEAN
    # ======================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # 2. DROP ID / USELESS COLUMNS
    # ======================================================
    id_keywords = ["id", "uuid", "invoice", "transaction", "row"]

    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue

        name = col.lower()

        if any(k in name for k in id_keywords):
            drop_cols.append(col)

    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ======================================================
    # 3. DROP LEAKAGE FEATURES (TARGET-DEPENDENT)
    # ======================================================
    if target_col.lower() in ["revenue", "sales", "profit", "weekly_sales"]:

        leakage_cols = [
            "Quantity",
            "Orders",
            "UnitPrice",
            "Avg_Order_Value",
            "Revenue",
            "Sales"
        ]

        df.drop(
            columns=[c for c in leakage_cols if c in df.columns and c != target_col],
            errors="ignore",
            inplace=True
        )

    # ======================================================
    # 4. DATE FEATURES
    # ======================================================
    date_col = None

    for col in df.columns:
        if col == target_col:
            continue

        try:
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.7:
                date_col = col

                df["year"]       = parsed.dt.year
                df["month"]      = parsed.dt.month
                df["week"]       = parsed.dt.isocalendar().week.astype(int)
                df["dayofweek"]  = parsed.dt.dayofweek
                df["is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)

                # cyclical encoding
                df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
                df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

                df.drop(columns=[col], inplace=True)
                break
        except:
            continue

    # ======================================================
    # 5. SORT (IMPORTANT FOR TIME SERIES)
    # ======================================================
    sort_cols = [c for c in ["year", "week", "month"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # ======================================================
    # 6. SAFE LAG FEATURES (ONLY IF TIME EXISTS)
    # ======================================================
    if date_col and target_col in df.columns:

        if len(df) > 50:  # only if enough data

            df["lag_7"]  = df[target_col].shift(7)
            df["lag_14"] = df[target_col].shift(14)

    # ======================================================
    # 7. REMOVE ROWS WITH NaN TARGET
    # ======================================================
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # ======================================================
    # 8. SPLIT
    # ======================================================
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    X = df.drop(columns=[target_col])

    # ======================================================
    # 9. ENCODE CATEGORICALS
    # ======================================================
    for col in X.select_dtypes(include="object").columns:
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
    # 11. REMOVE CONSTANT FEATURES
    # ======================================================
    if X.shape[1] > 0:
        selector = VarianceThreshold(threshold=0.0)
        X = pd.DataFrame(
            selector.fit_transform(X),
            columns=X.columns[selector.get_support()]
        )

    # ======================================================
    # 12. ALIGN FEATURES (FOR PREDICTION)
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X, y
