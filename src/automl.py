import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


# ======================================================
# 🎯 PROBLEM TYPE
# ======================================================
def detect_problem_type(y):
    return "classification" if y.nunique() <= 10 else "regression"


# ======================================================
# 🤖 TRAIN MODELS (FINAL VERSION)
# ======================================================
def train_models(X, y, problem_type, handle_imbalance=False):

    results = []

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if problem_type == "classification" else None
    )

    best_score = -np.inf
    best_model = None
    best_model_name = None
    best_metrics = {}

    # =========================
    # CV STRATEGY
    # =========================
    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # =========================
    # CLASS WEIGHT
    # =========================
    class_weight = None
    scale_pos_weight = 1.0
    catboost_weights = None

    if problem_type == "classification" and handle_imbalance:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        catboost_weights = [class_weight[c] for c in classes]

        if len(classes) == 2:
            neg = np.sum(y_train == classes[0])
            pos = np.sum(y_train == classes[1])
            scale_pos_weight = neg / max(pos, 1)

    # ======================================================
    # MODELS
    # ======================================================
    if problem_type == "classification":

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, class_weight=class_weight),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss", scale_pos_weight=scale_pos_weight),
            "LightGBM": LGBMClassifier(n_estimators=200, class_weight=class_weight),
            "CatBoost": CatBoostClassifier(iterations=200, verbose=False, class_weights=catboost_weights)
        }

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            # CV
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "F1": round(f1, 4),
                "CV Mean": round(cv_scores.mean(), 4),
                "CV Std": round(cv_scores.std(), 4)
            })

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name

                best_metrics = {
                    "accuracy": acc,
                    "f1": f1,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std()
                }

    else:

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=200),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(n_estimators=200),
            "LightGBM": LGBMRegressor(n_estimators=200),
            "CatBoost": CatBoostRegressor(iterations=200, verbose=False)
        }

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            # CV
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

            results.append({
                "Model": name,
                "R2": round(r2, 4),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "CV Mean": round(cv_scores.mean(), 4),
                "CV Std": round(cv_scores.std(), 4)
            })

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name

                best_metrics = {
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std()
                }

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs("models", exist_ok=True)

    if best_model is not None:
        joblib.dump(best_model, "models/best_model.pkl")
        joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    return pd.DataFrame(results), best_model_name, best_metrics
