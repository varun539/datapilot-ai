import pandas as pd
import numpy as np

def engineer_features(
    df,
    numeric_cols,
    date_cols,
    target_col,
    max_features=40
):
    """
    Safe feature engineering:
    - Squares + log only for real numeric signals
    - Limited interactions
    - Cyclic date features
    - HARD feature cap
    """

    engineered = pd.DataFrame(index=df.index)

    # -----------------------
    # Filter usable numeric columns
    # -----------------------
    safe_numeric = []
    for col in numeric_cols:
        name = col.lower()
        if "id" in name:
            continue
        if col == target_col:
            continue
        if df[col].nunique() < 5:   # avoid binary junk
            continue
        safe_numeric.append(col)

    # Limit base numeric count
    safe_numeric = safe_numeric[:6]   # <= VERY IMPORTANT

    # -----------------------
    # Numeric transforms
    # -----------------------
    for col in safe_numeric:
        engineered[f"{col}_sq"] = df[col] ** 2

        # log only if strictly positive
        if (df[col] > 0).all():
            engineered[f"log_{col}"] = np.log1p(df[col])

    # -----------------------
    # Pairwise interactions (LIMITED)
    # -----------------------
    for i in range(len(safe_numeric)):
        for j in range(i + 1, min(i + 3, len(safe_numeric))):
            c1 = safe_numeric[i]
            c2 = safe_numeric[j]
            engineered[f"{c1}_x_{c2}"] = df[c1] * df[c2]

    # -----------------------
    # Date features
    # -----------------------
    for date_col in date_cols:
        dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

        engineered["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        engineered["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)

    # -----------------------
    # Final cleanup
    # -----------------------
    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    engineered = engineered.fillna(0)

    # -----------------------
    # HARD FEATURE LIMIT
    # -----------------------
    if engineered.shape[1] > max_features:
        engineered = engineered.iloc[:, :max_features]

    return engineered
