import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
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
# 🤖 TRAIN MODELS (FINAL CLEAN VERSION)
# ======================================================
def train_models(X, y, problem_type, handle_imbalance=False):

    # 🔒 SAFETY
    if len(X) < 20:
        raise ValueError("Dataset too small for training")

    if y.nunique() <= 1:
        raise ValueError("Target has no variance")

    if X.shape[1] == 0:
        raise ValueError("No features after preprocessing")

    # =========================
    # SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if problem_type == "classification" else None
    )

    # =========================
    # CV
    # =========================
    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    best_score = -np.inf
    best_model = None
    best_name = None
    best_metrics = {}

    # ======================================================
    # MODELS
    # ======================================================
    if problem_type == "classification":

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42),
            "LightGBM": LGBMClassifier(n_estimators=200, random_state=42),
            "CatBoost": CatBoostClassifier(iterations=200, verbose=False, random_state=42)
        }

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            # ✅ CV ONLY ON TRAIN
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

                best_metrics = {
                    "holdout": {"accuracy": acc, "f1": f1},
                    "cv": {"mean": cv_scores.mean(), "std": cv_scores.std()}
                }

    else:

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(n_estimators=200, random_state=42),
            "LightGBM": LGBMRegressor(n_estimators=200, random_state=42),
            "CatBoost": CatBoostRegressor(iterations=200, verbose=False, random_state=42)
        }

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            # ✅ CV ONLY ON TRAIN
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

                best_metrics = {
                    "holdout": {"r2": r2, "mae": mae, "rmse": rmse},
                    "cv": {"mean": cv_scores.mean(), "std": cv_scores.std()}
                }

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(X.columns.tolist(), "models/features.pkl")
    joblib.dump(best_metrics, "models/metrics.pkl")

    return best_model, best_name, best_metrics
