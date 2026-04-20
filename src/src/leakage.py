import pandas as pd
import numpy as np


def detect_leakage(X, y, threshold=0.9):
    """
    Detects potential data leakage based on correlation.

    Returns:
    - warnings (list of strings)
    - high_risk (bool)
    """

    warnings = []
    high_risk = False

    if X.empty or y.empty:
        return ["Dataset is empty"], True

    # Combine safely
    df = X.copy()
    df["_target_"] = y

    # Correlation
    corr = df.corr(numeric_only=True)

    if "_target_" not in corr.columns:
        return ["Unable to compute correlations"], True

    target_corr = corr["_target_"].drop("_target_")

    # Check high correlations
    for col, val in target_corr.items():

        if abs(val) > threshold:
            warnings.append(
                f"🚨 HIGH RISK: '{col}' is highly correlated with target ({val:.2f})"
            )
            high_risk = True

        elif abs(val) > 0.75:
            warnings.append(
                f"⚠️ Suspicious: '{col}' has strong correlation ({val:.2f})"
            )

    # Check duplicate columns
    duplicates = X.T.duplicated()
    if duplicates.any():
        dup_cols = X.columns[duplicates].tolist()
        warnings.append(f"⚠️ Duplicate features detected: {dup_cols}")

    # Check constant columns
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        warnings.append(f"⚠️ Constant features: {constant_cols}")

    if not warnings:
        warnings.append("✅ No major leakage detected")

    return warnings, high_risk
