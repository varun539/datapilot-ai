import pandas as pd
import numpy as np


def prepare_features(df, profile, target_col, training=True):

    df = df.copy()

    # =========================
    # CLEAN
    # =========================
    df = df.replace([np.inf, -np.inf], np.nan)

    # =========================
    # DATE FEATURES
    # =========================
    date_col = None

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")

                if parsed.notna().mean() > 0.7:
                    date_col = col

                    df[col] = parsed
                    df["year"] = parsed.dt.year
                    df["month"] = parsed.dt.month
                    df["week"] = parsed.dt.isocalendar().week.astype(int)
                    df["dayofweek"] = parsed.dt.dayofweek

                    break
            except:
                pass

    # =========================
    # SORT (🔥 VERY IMPORTANT)
    # =========================
    if "Store" in df.columns and date_col is not None:
        df = df.sort_values(["Store", date_col])

    # =========================
    # STORE FEATURES (BIG BOOST)
    # =========================
    if "Store" in df.columns and target_col in df.columns:
        df["store_avg"] = df.groupby("Store")[target_col].transform("mean")
        df["store_std"] = df.groupby("Store")[target_col].transform("std")

    # =========================
    # LAG FEATURES (TIME INTELLIGENCE)
    # =========================
    if "Store" in df.columns and target_col in df.columns:
        df["lag_1"] = df.groupby("Store")[target_col].shift(1)
        df["lag_2"] = df.groupby("Store")[target_col].shift(2)
        df["lag_4"] = df.groupby("Store")[target_col].shift(4)

    # =========================
    # ROLLING FEATURES
    # =========================
    if "Store" in df.columns and target_col in df.columns:
        df["rolling_mean_4"] = df.groupby("Store")[target_col].shift(1).rolling(4).mean()
        df["rolling_std_4"] = df.groupby("Store")[target_col].shift(1).rolling(4).std()

    # =========================
    # DROP NA AFTER LAG (IMPORTANT)
    # =========================

# =========================
    # DROP NA AFTER LAG
    # =========================
    df = df.dropna()
    
    # =========================
    # SPLIT X AND y
    # =========================
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # =========================
    # FINAL CLEAN
    # =========================
    X = X.fillna(0)
    
    return X, y
    
