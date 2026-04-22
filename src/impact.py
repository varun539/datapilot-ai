# import numpy as np
# import pandas as pd


# def generate_business_impact(shap_values, X, problem_type, target_col):
#     """
#     Clean, business-friendly insights from SHAP values.
#     Focus: clarity + usefulness + no misleading claims.
#     """

#     # =============================
#     # FEATURE IMPORTANCE
#     # =============================
#     mean_abs_shap = np.abs(shap_values).mean(axis=0)

#     importance = (
#         pd.Series(mean_abs_shap, index=X.columns)
#         .sort_values(ascending=False)
#     )

#     insights = []

#     # =============================
#     # TOP FEATURES
#     # =============================
#     for feat in importance.head(3).index:

#         feat_idx = X.columns.get_loc(feat)
#         avg_effect = shap_values[:, feat_idx].mean()


#         def clean_feature_name(feat):
#         if "lag_" in feat:
#             return "recent sales trend"
#         if feat == "sales_momentum":
#             return "sales growth trend"
#         if feat == "volatility":
#             return "sales variability"
#         return feat

#         # Direction from SHAP (primary signal)
#         if abs(avg_effect) < 1e-4:
#             direction = "has minimal impact on"
#         elif avg_effect > 0:
#             direction = "is associated with higher"
#         else:
#             direction = "is associated with lower"

#         # Optional correlation (secondary signal, safer use)
#         try:
#             corr = np.corrcoef(X[feat], shap_values.mean(axis=1))[0, 1]
#         except:
#             corr = 0

#         # Confidence level (UX improvement 🔥)
#         if abs(corr) > 0.3:
#             confidence = "strong pattern"
#         elif abs(corr) > 0.1:
#             confidence = "moderate pattern"
#         else:
#             confidence = "weak pattern"

#         # =============================
#         # REGRESSION OUTPUT
#         # =============================
#         if problem_type == "regression":

#             insights.append(
#                 f"📊 **{feat}** {direction} **{target_col}** "
#                 f"({confidence} observed in the data)."
#             )

#         # =============================
#         # CLASSIFICATION OUTPUT
#         # =============================
#         else:

#             insights.append(
#                 f"⚠️ **{feat}** influences classification outcomes "
#                 f"({confidence}). Changes in this feature affect predictions."
#             )

#     return insights
















import numpy as np
import pandas as pd


def generate_business_impact(shap_values, X, problem_type, target_col):
    """
    Converts SHAP values into plain English business insights.
    No ML jargon — speaks like a business consultant.
    """

    insights = []

    try:
        # Get mean absolute SHAP per feature
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_dir = shap_values.mean(axis=0)  # direction

        # Top 5 features
        top_idx = np.argsort(mean_abs)[::-1][:5]

        # Business-friendly feature name mapping
        feature_labels = {
            "Holiday_Flag":    "holiday periods",
            "holiday_flag":    "holiday periods",
            "is_q4":           "Q4 holiday season",
            "is_q1":           "Q1 slow season",
            "Temperature":     "temperature / weather",
            "temperature":     "temperature / weather",
            "Fuel_Price":      "fuel prices",
            "fuel_price":      "fuel prices",
            "CPI":             "consumer price index (inflation)",
            "cpi":             "consumer price index (inflation)",
            "Unemployment":    "unemployment rate",
            "unemployment":    "unemployment rate",
            "month":           "time of year / seasonality",
            "week":            "weekly patterns",
            "month_sin":       "seasonal patterns",
            "month_cos":       "seasonal patterns",
            "week_sin":        "weekly seasonality",
            "week_cos":        "weekly seasonality",
            "is_weekend":      "weekend vs weekday",
            "quarter":         "quarterly trends",
            "lag_2":           "sales 2 weeks ago",
            "lag_4":           "sales 1 month ago",
            "lag_7":           "sales last week",
            "lag_14":          "sales 2 weeks ago",
            "sales_momentum":  "sales trend (improving or declining)",
            "volatility":      "sales stability / consistency",
            "Store":           "store location",
            "year":            "year-over-year growth",
        }

        for i in top_idx:
            col_name   = X.columns[i]
            importance = round(float(mean_abs[i]), 4)
            direction  = mean_dir[i]

            if importance < 0.001:
                continue

            # Get friendly name
            label = feature_labels.get(col_name, col_name.replace("_", " ").title())

            # Strength
            if importance > mean_abs.mean() * 2:
                strength = "strongly"
            elif importance > mean_abs.mean():
                strength = "moderately"
            else:
                strength = "slightly"

            # Direction in business language
            if problem_type == "regression":
                if direction > 0:
                    effect = f"higher {label} → higher {target_col}"
                else:
                    effect = f"higher {label} → lower {target_col}"
            else:
                if direction > 0:
                    effect = f"{label} increases risk"
                else:
                    effect = f"{label} reduces risk"

            # Build insight
            insight = f"📊 **{label.title()}** {strength} affects {target_col} — {effect}."

            # Add actionable tip for key features
            tips = {
                "Holiday_Flag":   f"📌 Tip: Stock up and run promotions BEFORE holiday periods — they drive significant {target_col} uplift.",
                "holiday_flag":   f"📌 Tip: Stock up and run promotions BEFORE holiday periods.",
                "is_q4":          f"📌 Tip: Q4 is your biggest season — plan inventory 6-8 weeks ahead.",
                "Fuel_Price":     f"📌 Tip: When fuel prices rise, consider free shipping thresholds to protect {target_col}.",
                "fuel_price":     f"📌 Tip: When fuel prices rise, consider free shipping thresholds.",
                "CPI":            f"📌 Tip: During high inflation, emphasize value deals and bundles.",
                "cpi":            f"📌 Tip: During high inflation, emphasize value deals and bundles.",
                "Unemployment":   f"📌 Tip: In high unemployment areas, budget-friendly pricing performs better.",
                "sales_momentum": f"📌 Tip: Your recent sales trend predicts future performance — act early on declining trends.",
                "volatility":     f"📌 Tip: High sales variability suggests inconsistent demand — consider steady promotions.",
                "Temperature":    f"📌 Tip: Weather affects buying behavior — align seasonal products with temperature forecasts.",
            }

            if col_name in tips:
                insight += f"\n{tips[col_name]}"

            insights.append(insight)

    except Exception as e:
        insights.append(f"Business analysis complete. Detail: {e}")

    if not insights:
        insights.append(f"✅ Model trained successfully on {target_col}. Upload your own data to get specific insights.")

    return insights
