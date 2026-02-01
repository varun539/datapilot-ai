import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


# ======================================================
# 🕒 AUTO MODE DETECTION
# ======================================================
def detect_training_mode(df, target_col, profile):
    """
    Detect whether dataset should be treated as:
    - Time Series
    - Standard ML
    """

    datetime_cols = profile.get("datetime_cols", [])

    # No datetime column → normal ML
    if not datetime_cols:
        return "standard"

    # If target changes over time → time series
    for date_col in datetime_cols:
        if date_col in df.columns and target_col in df.columns:
            temp = df[[date_col, target_col]].dropna()

            # Must have enough unique dates
            if temp[date_col].nunique() > 10:
                return "time_series"

    return "standard"

# ======================================================
# Detect Problem Type
# ======================================================
def detect_problem_type(y):
    if y.nunique() <= 10:
        return "classification"
    else:
        return "regression"


# ======================================================
# Train Models (🔥 WITH CLASS WEIGHT SUPPORT)
# ======================================================
def train_models(X, y, problem_type, handle_imbalance=False):
    results = []

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if problem_type == "classification" else None
    )

    best_score = -999
    best_model = None
    best_model_name = None

    # ==================================================
    # ⚖️ CLASS WEIGHT CALCULATION
    # ==================================================
    class_weight = None
    scale_pos_weight = 1.0
    catboost_weights = None

    if problem_type == "classification" and handle_imbalance:
        classes = np.unique(y_train)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )

        class_weight = dict(zip(classes, weights))

        # For CatBoost → list format
        catboost_weights = [class_weight[c] for c in classes]

        # For XGBoost → negative / positive ratio
        if len(classes) == 2:
            neg = np.sum(y_train == classes[0])
            pos = np.sum(y_train == classes[1])
            scale_pos_weight = neg / max(pos, 1)

        print("✅ Class weights applied:", class_weight)

    # ------------------------------
    # CLASSIFICATION
    # ------------------------------
    if problem_type == "classification":

        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                class_weight=class_weight,
                random_state=42
            ),

            "Gradient Boosting": GradientBoostingClassifier(),

            "XGBoost": XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                random_state=42
            ),

            "LightGBM": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                class_weight=class_weight,
                random_state=42
            ),

            "CatBoost": CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                verbose=False,
                class_weights=catboost_weights
            )
        }

        best_score = -1

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, average="weighted", zero_division=0)
            recall = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4)
            })

            if acc > best_score:
                best_score = acc
                best_model = model
                best_model_name = name

    # ------------------------------
    # REGRESSION
    # ------------------------------
    else:

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(),

            "XGBoost": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),

            "LightGBM": LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                random_state=42
            ),

            "CatBoost": CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                verbose=False
            )
        }

        best_score = -999

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            results.append({
                "Model": name,
                "R2": round(r2, 4),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2)
            })

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name

    # ------------------------------
    # Save Best Model
    # ------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    results_df = pd.DataFrame(results)
    return results_df, best_model_name


# ======================================================
# 🔥 Hyperparameter Tuning
# ======================================================
def tune_best_model(model, X, y, problem_type):

    if problem_type == "regression":
        param_grid = {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [3, 5, 7, None],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0]
        }
        scoring = "r2"

    else:
        param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0]
        }
        scoring = "accuracy"

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=15,
        scoring=scoring,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)

    return (
        search.best_estimator_,
        search.best_params_,
        round(search.best_score_, 4)
    )

# ======================================================
# ⚠️ Class Imbalance Detector
# ======================================================
def detect_class_imbalance(y, threshold=0.75):
    """
    Returns (is_imbalanced, majority_ratio)
    """
    value_ratios = y.value_counts(normalize=True)
    majority_ratio = value_ratios.iloc[0]

    if majority_ratio >= threshold:
        return True, round(majority_ratio, 3)

    return False, round(majority_ratio, 3)



# ======================================================
# ⚠️ DATA LEAKAGE GUARD
# ======================================================
def detect_data_leakage(X, y, threshold=0.98):
    """
    Detects potential data leakage using correlation with target.
    Returns:
        (is_leakage, leaked_features)
    """

    leakage_features = []

    # Only numeric features are checked
    X_numeric = X.select_dtypes(include=["number"])

    for col in X_numeric.columns:
        try:
            corr = np.corrcoef(X_numeric[col], y)[0, 1]
            if abs(corr) >= threshold:
                leakage_features.append((col, round(corr, 4)))
        except Exception:
            continue

    if leakage_features:
        return True, leakage_features

    return False, []

def generate_business_impact(shap_values, X, problem_type, target_col):
    """
    Returns human-readable business insights
    """

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(
        mean_abs_shap, index=X.columns
    ).sort_values(ascending=False)

    insights = []

    for feat in feature_importance.head(3).index:
        corr = np.corrcoef(X[feat], X.mean(axis=1))[0, 1]

        direction = "increase" if corr > 0 else "decrease"

        if problem_type == "regression":
            insights.append(
                f"📈 Changes in **{feat}** are strongly associated with "
                f"an **{direction} in {target_col}**, suggesting this feature "
                f"has high business impact."
            )
        else:
            insights.append(
                f"⚠️ **{feat}** significantly influences classification outcomes "
                f"and should be monitored for risk reduction."
            )

    return insights



