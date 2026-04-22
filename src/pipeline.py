# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import VarianceThreshold


# def prepare_features(df, profile, target_col, training=True, feature_schema=None):
#     """
#     🔥 FINAL PIPELINE — NO LEAKAGE, REAL BUSINESS LOGIC

#     ✔ Removes target-derived features
#     ✔ Removes useless IDs & constants
#     ✔ Keeps meaningful time patterns
#     ✔ Safe lag features (ONLY past info)
#     ✔ Recruiter-level clean pipeline
#     """

#     df = df.copy()

#     # ======================================================
#     # 1. CLEAN
#     # ======================================================
#     df = df.replace([np.inf, -np.inf], np.nan)

#     # ======================================================
#     # 2. DROP ID / USELESS COLUMNS
#     # ======================================================
#     id_keywords = ["id", "uuid", "invoice", "transaction", "row"]

#     drop_cols = []
#     for col in df.columns:
#         if col == target_col:
#             continue

#         name = col.lower()

#         if any(k in name for k in id_keywords):
#             drop_cols.append(col)

#     df.drop(columns=drop_cols, errors="ignore", inplace=True)

#     # ======================================================
#     # 3. DROP LEAKAGE FEATURES (TARGET-DEPENDENT)
#     # ======================================================
#     if target_col.lower() in ["revenue", "sales", "profit", "weekly_sales"]:

#         leakage_cols = [
#             "Quantity",
#             "Orders",
#             "UnitPrice",
#             "Avg_Order_Value",
#             "Revenue",
#             "Sales"
#         ]

#         df.drop(
#             columns=[c for c in leakage_cols if c in df.columns and c != target_col],
#             errors="ignore",
#             inplace=True
#         )

#     # ======================================================
#     # 4. DATE FEATURES
#     # ======================================================
#     date_col = None

#     for col in df.columns:
#         if col == target_col:
#             continue

#         try:
#             parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
#             if parsed.notna().mean() > 0.7:
#                 date_col = col

#                 df["year"]       = parsed.dt.year
#                 df["month"]      = parsed.dt.month
#                 df["week"]       = parsed.dt.isocalendar().week.astype(int)
#                 df["dayofweek"]  = parsed.dt.dayofweek
#                 df["is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)

#                 # cyclical encoding
#                 df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
#                 df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

#                 df.drop(columns=[col], inplace=True)
#                 break
#         except:
#             continue

#     # ======================================================
#     # 5. SORT (IMPORTANT FOR TIME SERIES)
#     # ======================================================
#     sort_cols = [c for c in ["year", "week", "month"] if c in df.columns]
#     if sort_cols:
#         df = df.sort_values(sort_cols).reset_index(drop=True)

#     # ======================================================
#     # 6. SAFE LAG FEATURES (ONLY IF TIME EXISTS)
#     # ======================================================
#     if date_col and target_col in df.columns:

#         if len(df) > 50:  # only if enough data

#             df["lag_7"]  = df[target_col].shift(7)
#             df["lag_14"] = df[target_col].shift(14)

#     # ======================================================
#     # 7. REMOVE ROWS WITH NaN TARGET
#     # ======================================================
#     df = df.dropna(subset=[target_col]).reset_index(drop=True)

#     # ======================================================
#     # 8. SPLIT
#     # ======================================================
#     y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
#     X = df.drop(columns=[target_col])

#     # ======================================================
#     # 9. ENCODE CATEGORICALS
#     # ======================================================
#     for col in X.select_dtypes(include="object").columns:
#         if X[col].nunique() <= 20:
#             X = pd.get_dummies(X, columns=[col], drop_first=True)
#         else:
#             X.drop(columns=[col], inplace=True)

#     # ======================================================
#     # 10. FINAL CLEAN
#     # ======================================================
#     X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
#     X = X.replace([np.inf, -np.inf], 0)

#     # ======================================================
#     # 11. REMOVE CONSTANT FEATURES
#     # ======================================================
#     if X.shape[1] > 0:
#         selector = VarianceThreshold(threshold=0.0)
#         X = pd.DataFrame(
#             selector.fit_transform(X),
#             columns=X.columns[selector.get_support()]
#         )

#     # ======================================================
#     # 12. ALIGN FEATURES (FOR PREDICTION)
#     # ======================================================
#     if not training and feature_schema is not None:
#         for col in feature_schema:
#             if col not in X.columns:
#                 X[col] = 0
#         X = X[feature_schema]

#     return X, y











import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def prepare_features(df, profile, target_col, training=True, feature_schema=None):
    """
    PRODUCTION PIPELINE — Zero Leakage, Real Business Logic
    - Daily data   → lag_7, lag_14
    - Weekly data  → lag_2, lag_4
    - Auto-detects frequency from date range vs row count
    """

    df = df.copy()

    # ======================================================
    # 1. CLEAN
    # ======================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # 2. DROP PURE ID COLUMNS
    # ======================================================
    id_keywords = ["uuid", "invoice_no", "transaction_id", "row_id"]
    drop_cols = [
        c for c in df.columns
        if c != target_col and c.lower() in id_keywords
    ]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ======================================================
    # 3. DROP TARGET-DERIVED LEAKY COLUMNS
    # ======================================================
    leaky = [
        "Avg_Order_Value", "avg_order_value",
        "revenue_per_order", "sales_per_order"
    ]
    df.drop(
        columns=[c for c in leaky if c in df.columns and c != target_col],
        errors="ignore", inplace=True
    )

    # ======================================================
    # 4. DATE FEATURES + DETECT FREQUENCY
    # ======================================================
    date_col  = None
    is_weekly = False

    for col in list(df.columns):
        if col == target_col:
            continue
        try:
            parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            if parsed.notna().mean() > 0.7:
                date_col = col

                df["year"]       = parsed.dt.year
                df["month"]      = parsed.dt.month
                df["week"]       = parsed.dt.isocalendar().week.astype(int)
                df["quarter"]    = parsed.dt.quarter
                df["dayofweek"]  = parsed.dt.dayofweek
                df["is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
                df["is_q4"]      = (parsed.dt.quarter == 4).astype(int)
                df["is_q1"]      = (parsed.dt.quarter == 1).astype(int)

                # Cyclic encoding
                df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
                df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
                df["week_sin"]  = np.sin(2 * np.pi * df["week"]  / 52)
                df["week_cos"]  = np.cos(2 * np.pi * df["week"]  / 52)

                # Detect weekly vs daily
                date_range_days = max((parsed.max() - parsed.min()).days, 1)
                n_rows = len(df)
                is_weekly = (date_range_days > 365 and n_rows < 500)

                df.drop(columns=[col], inplace=True)
                break
        except Exception:
            continue

    # ======================================================
    # 5. SORT BY TIME (with Store grouping if exists)
    # ======================================================
    sort_cols = [c for c in ["year", "week", "month"] if c in df.columns]

    if "Store" in df.columns and sort_cols:
        df = df.sort_values(["Store"] + sort_cols).reset_index(drop=True)
    elif sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # ======================================================
    # 6. SMART LAG FEATURES — no leakage
    # ======================================================
    if date_col and target_col in df.columns and len(df) > 30:

        group_col = "Store" if "Store" in df.columns else None

        # Weekly → lag_2 (2wk), lag_4 (1mo)
        # Daily  → lag_7 (1wk), lag_14 (2wk)
        lag_a, lag_b = (2, 4) if is_weekly else (7, 14)

        if group_col:
            grp = df.groupby(group_col)[target_col]
            df[f"lag_{lag_a}"]     = grp.shift(lag_a)
            df[f"lag_{lag_b}"]     = grp.shift(lag_b)
            df["sales_momentum"]   = df[f"lag_{lag_a}"] - df[f"lag_{lag_b}"]
            df["volatility"]       = (
                grp.shift(lag_a)
                .rolling(lag_a, min_periods=2)
                .std()
                .fillna(0)
                .reset_index(level=0, drop=True)
            )
        else:
            df[f"lag_{lag_a}"]   = df[target_col].shift(lag_a)
            df[f"lag_{lag_b}"]   = df[target_col].shift(lag_b)
            df["sales_momentum"] = df[f"lag_{lag_a}"] - df[f"lag_{lag_b}"]
            df["volatility"]     = (
                df[target_col]
                .shift(lag_a)
                .rolling(lag_a, min_periods=2)
                .std()
                .fillna(0)
            )

    # ======================================================
    # 7. DROP NaN
    # ======================================================
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # ======================================================
    # 8. SPLIT X, y
    # ======================================================
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    X = df.drop(columns=[target_col])

    # ======================================================
    # 9. ENCODE CATEGORICALS
    # ======================================================
    for col in X.select_dtypes(include="object").columns.tolist():
        if X[col].nunique() <= 20:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X.drop(columns=[col], inplace=True)

    # ======================================================
    # 10. FINAL CLEAN
    # ======================================================
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Remove zero-variance columns
    if X.shape[1] > 0:
        try:
            sel   = VarianceThreshold(threshold=0.0)
            X_arr = sel.fit_transform(X)
            X     = pd.DataFrame(X_arr, columns=X.columns[sel.get_support()])
        except Exception:
            pass

    # ======================================================
    # 11. SCHEMA ALIGNMENT (prediction only)
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X, y
