# import pandas as pd
# import numpy as np


# # ======================================================
# # 🎯 MAIN FEATURE PIPELINE
# # ======================================================
# def prepare_features(df, profile, target_col, training=True, feature_schema=None):
#     """
#     Business-ready feature pipeline

#     ✔ No leakage
#     ✔ Stable
#     ✔ Works for retail / generic datasets
#     """

#     df = df.copy()

#     # ======================================================
#     # 1. CLEAN
#     # ======================================================
#     df = df.replace([np.inf, -np.inf], np.nan)

#     # ======================================================
#     # 2. TARGET VALIDATION
#     # ======================================================
#     if target_col not in df.columns:
#         return pd.DataFrame(), pd.Series()

#     df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
#     df = df.dropna(subset=[target_col])

#     if df[target_col].nunique() <= 1:
#         return pd.DataFrame(), pd.Series()

#     # ======================================================
#     # 3. DROP ID / LEAKAGE COLUMNS
#     # ======================================================
#     drop_keywords = ["id", "uuid", "invoice", "transaction", "row", "index"]

#     drop_cols = []

#     for col in df.columns:
#         if col == target_col:
#             continue

#         name = col.lower()

#         if any(k in name for k in drop_keywords):
#             drop_cols.append(col)

#         if name == "store":
#             drop_cols.append(col)

#     df.drop(columns=drop_cols, errors="ignore", inplace=True)

#     # ======================================================
#     # 4. DATE FEATURES
#     # ======================================================
#     date_col = None

#     for col in df.columns:
#         if col != target_col and "date" in col.lower():
#             date_col = col
#             break

#     if date_col:
#         parsed = pd.to_datetime(df[date_col], errors="coerce")

#         if parsed.notna().mean() > 0.7:
#             df["year"] = parsed.dt.year
#             df["month"] = parsed.dt.month
#             df["dayofweek"] = parsed.dt.dayofweek
#             df["week"] = parsed.dt.isocalendar().week.astype(int)
#             df["is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)

#             # seasonality
#             df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
#             df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

#             df.drop(columns=[date_col], inplace=True)

#     # ======================================================
#     # 5. SPLIT
#     # ======================================================
#     y = df[target_col]
#     X = df.drop(columns=[target_col])

#     # ======================================================
#     # 6. BUSINESS FEATURES
#     # ======================================================
#     business_cols = ["Revenue", "Orders", "Quantity", "Avg_Order_Value"]

#     for col in business_cols:
#         if col in X.columns:
#             X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))

#     # ======================================================
#     # 7. INTERACTIONS (LIMITED)
#     # ======================================================
#     important = [c for c in ["Orders", "Quantity", "Avg_Order_Value"] if c in X.columns]

#     if len(important) >= 2:
#         for i in range(len(important)):
#             for j in range(i + 1, len(important)):
#                 c1 = important[i]
#                 c2 = important[j]
#                 X[f"{c1}_x_{c2}"] = X[c1] * X[c2]

#     # ======================================================
#     # 8. NUMERIC CLEAN
#     # ======================================================
#     for col in X.select_dtypes(include=np.number).columns:
#         X[col] = pd.to_numeric(X[col], errors="coerce")

#     # ======================================================
#     # 9. CATEGORICAL ENCODING
#     # ======================================================
#     cat_cols = X.select_dtypes(include="object").columns.tolist()

#     for col in cat_cols:
#         if X[col].nunique() <= 20:
#             X = pd.get_dummies(X, columns=[col], drop_first=True)
#         else:
#             X.drop(columns=[col], inplace=True)

#     # ======================================================
#     # 10. FINAL CLEAN
#     # ======================================================
#     X = X.fillna(0)
#     X = X.replace([np.inf, -np.inf], 0)

#     # ======================================================
#     # 11. LOW VARIANCE FILTER
#     # ======================================================
#     from sklearn.feature_selection import VarianceThreshold

#     try:
#         selector = VarianceThreshold(threshold=0.0)
#         X = pd.DataFrame(
#             selector.fit_transform(X),
#             columns=X.columns[selector.get_support()]
#         )
#     except:
#         pass  # safe fallback

#     # ======================================================
#     # 12. SCHEMA ALIGNMENT
#     # ======================================================
#     if training:
#         feature_schema = X.columns.tolist()
#     else:
#         if feature_schema is not None:
#             for col in feature_schema:
#                 if col not in X.columns:
#                     X[col] = 0
#             X = X[feature_schema]

#     return X, y










import pandas as pd
import numpy as np


def prepare_features(df, profile, target_col, training=True, feature_schema=None):
    """
    ZERO LEAKAGE pipeline for real business use.

    Key rules:
    - NO Avg_Order_Value (derived from target)
    - NO Store column (encodes target history)
    - NO lag_1 or lag_2 (too correlated)
    - Safe lags only: 7+ days or 4+ weeks back
    """

    df = df.copy()

    # ======================================================
    # 1. CLEAN
    # ======================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # 2. DROP LEAKY COLUMNS
    # ======================================================
    always_drop = [
        # Derived from target = leakage
        "Avg_Order_Value", "avg_order_value",
        "revenue_per_order", "sales_per_order",
        "revenue_per_customer",
        # Store encodes avg sales per location = leakage
        "Store", "store",
        # Pure IDs
        "uuid", "invoice_id", "order_id",
        "record_id", "customer_id", "row_id"
    ]

    drop_cols = [
        c for c in df.columns
        if c != target_col and (
            c in always_drop or
            c.lower() in [x.lower() for x in always_drop]
        )
    ]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ======================================================
    # 3. DATE FEATURES
    # ======================================================
    date_col = None
    for col in list(df.columns):
        if col == target_col:
            continue
        if df[col].dtype == "object":
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
                    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
                    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
                    df["week_sin"]   = np.sin(2 * np.pi * df["week"]  / 52)
                    df["week_cos"]   = np.cos(2 * np.pi * df["week"]  / 52)
                    df.drop(columns=[col], inplace=True)
                    break
            except Exception:
                pass

    # ======================================================
    # 4. SORT BY TIME
    # ======================================================
    sort_cols = [c for c in ["year", "week", "month"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # ======================================================
    # 5. SAFE LAG FEATURES — adapted to dataset size
    # ======================================================
    if target_col in df.columns and date_col:
        n = len(df)

        # Daily data (>200 rows) → 7/14 day lags
        # Weekly data (<200 rows) → 4/8 week lags
        lag_a = 7  if n > 200 else 4
        lag_b = 14 if n > 200 else 8

        df[f"lag_{lag_a}"] = df[target_col].shift(lag_a)
        df[f"lag_{lag_b}"] = df[target_col].shift(lag_b)
        df["sales_momentum"] = df[f"lag_{lag_a}"] - df[f"lag_{lag_b}"]
        df["volatility"] = (
            df[target_col]
            .shift(lag_a)
            .rolling(lag_a, min_periods=2)
            .std()
            .fillna(0)
        )

    # ======================================================
    # 6. DROP NaN
    # ======================================================
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # ======================================================
    # 7. SPLIT
    # ======================================================
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    X = df.drop(columns=[target_col])

    # ======================================================
    # 8. ENCODE CATEGORICALS
    # ======================================================
    for col in X.select_dtypes(include="object").columns.tolist():
        if X[col].nunique() <= 20:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X.drop(columns=[col], inplace=True)

    # ======================================================
    # 9. FINAL CLEAN
    # ======================================================
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # ======================================================
    # 10. SCHEMA ALIGNMENT (prediction only)
    # ======================================================
    if not training and feature_schema is not None:
        for col in feature_schema:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_schema]

    return X, y
