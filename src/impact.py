import numpy as np
import pandas as pd


def generate_business_impact(shap_values, X, problem_type, target_col):
    """
    Generate professional, business-safe insights from SHAP values.
    Focus: statistical patterns, not causal claims.
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

        # Direction from SHAP
        feat_idx = X.columns.get_loc(feat)
        avg_effect = shap_values[:, feat_idx].mean()

        # Correlation (extra robustness 🔥)
        try:
            corr = np.corrcoef(X[feat], shap_values.mean(axis=1))[0, 1]
        except:
            corr = 0

        # =============================
        # INTERPRET DIRECTION SMARTLY
        # =============================
        if abs(corr) < 0.1:
            relation = "no strong directional relationship"
        elif corr > 0:
            relation = "a positive association"
        else:
            relation = "a negative association"

        # =============================
        # REGRESSION INSIGHTS
        # =============================
        if problem_type == "regression":

            insights.append(
                f"📊 **{feat}** shows {relation} with **{target_col}** "
                f"in this dataset. This suggests that variations in this feature "
                f"are linked to changes in {target_col}, although the relationship "
                f"may be influenced by external or contextual factors."
            )

        # =============================
        # CLASSIFICATION INSIGHTS
        # =============================
        else:

            insights.append(
                f"⚠️ **{feat}** is a key driver in classification outcomes. "
                f"Changes in this feature are associated with shifts in predicted "
                f"classes and may impact model decisions."
            )

    return insights
