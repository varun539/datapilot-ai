import numpy as np
import pandas as pd


def generate_business_impact(shap_values, X, problem_type, target_col):
    """
    Clean, business-friendly insights from SHAP values.
    Focus: clarity + usefulness + no misleading claims.
    """

    # =============================
    # FEATURE IMPORTANCE
    # =============================
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = (
        pd.Series(mean_abs_shap, index=X.columns)
        .sort_values(ascending=False)
    )

    insights = []

    # =============================
    # TOP FEATURES
    # =============================
    for feat in importance.head(3).index:

        feat_idx = X.columns.get_loc(feat)
        avg_effect = shap_values[:, feat_idx].mean()


        def clean_feature_name(feat):
        if "lag_" in feat:
            return "recent sales trend"
        if feat == "sales_momentum":
            return "sales growth trend"
        if feat == "volatility":
            return "sales variability"
        return feat

        # Direction from SHAP (primary signal)
        if abs(avg_effect) < 1e-4:
            direction = "has minimal impact on"
        elif avg_effect > 0:
            direction = "is associated with higher"
        else:
            direction = "is associated with lower"

        # Optional correlation (secondary signal, safer use)
        try:
            corr = np.corrcoef(X[feat], shap_values.mean(axis=1))[0, 1]
        except:
            corr = 0

        # Confidence level (UX improvement 🔥)
        if abs(corr) > 0.3:
            confidence = "strong pattern"
        elif abs(corr) > 0.1:
            confidence = "moderate pattern"
        else:
            confidence = "weak pattern"

        # =============================
        # REGRESSION OUTPUT
        # =============================
        if problem_type == "regression":

            insights.append(
                f"📊 **{feat}** {direction} **{target_col}** "
                f"({confidence} observed in the data)."
            )

        # =============================
        # CLASSIFICATION OUTPUT
        # =============================
        else:

            insights.append(
                f"⚠️ **{feat}** influences classification outcomes "
                f"({confidence}). Changes in this feature affect predictions."
            )

    return insights
