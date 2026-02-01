import numpy as np
import pandas as pd

def generate_business_impact(shap_values, X, problem_type, target_col):
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)

    insights = []

    for feat in importance.head(3).index:
        direction = "increase" if shap_values[:, X.columns.get_loc(feat)].mean() > 0 else "decrease"

        if problem_type == "regression":
            insights.append(
                f"📈 **{feat}** strongly impacts **{target_col}**. "
                f"Increasing this feature tends to **{direction} predictions**, "
                f"making it a key business lever."
            )
        else:
            insights.append(
                f"⚠️ **{feat}** significantly affects classification outcomes "
                f"and should be closely monitored."
            )

    return insights
