import numpy as np
import pandas as pd


def generate_business_impact(shap_values, X, problem_type, target_col):
    """
    Generate business-safe, human-readable insights from SHAP values.
    NOTE: Insights reflect correlation, not causation.
    """

    # Mean absolute SHAP importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.Series(mean_abs_shap, index=X.columns)
        .sort_values(ascending=False)
    )

    insights = []

    for feat in importance.head(3).index:

        # Direction based on average SHAP contribution
        avg_effect = shap_values[:, X.columns.get_loc(feat)].mean()

        if avg_effect > 0:
            direction_text = "higher values are associated with higher"
        else:
            direction_text = "higher values are associated with lower"

        if problem_type == "regression":
            insights.append(
                f"📊 **{feat}** shows a strong statistical association with "
                f"**{target_col}**. Historically, {direction_text} "
                f"{target_col} values. "
                f"This relationship may be influenced by external or economic factors "
                f"and does not imply causation."
            )
        else:
            insights.append(
                f"⚠️ **{feat}** has a strong influence on classification outcomes. "
                f"Changes in this feature are associated with shifts in predicted "
                f"classes and should be monitored as part of risk analysis."
            )

    return insights
