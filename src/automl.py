import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


# ======================================================
# 🎯 PROBLEM TYPE
# ======================================================
def detect_problem_type(y):
    return "classification" if y.nunique() <= 10 else "regression"


# ======================================================
# 🔒 SORT HELPER (CRITICAL FIX)
# ======================================================
def ensure_time_order(X, y):
    """
    Ensures data is sorted (prevents leakage in time-series CV)
    """
    # If index already sequential → assume safe
    if isinstance(X.index, pd.RangeIndex):
        return X, y

    # Otherwise sort by index
    order = np.argsort(X.index.values)
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)

    return X, y


# ======================================================
# 🤖 TRAIN MODELS (FINAL SAFE VERSION)
# ======================================================
def train_models(X, y, problem_type):

    # =========================
    # SAFETY CHECKS
    # =========================
    if len(X) < 30:
        raise ValueError("Dataset too small — need at least 30 rows")

    if y.nunique() <= 1:
        raise ValueError("Target has no variance")

    if X.shape[1] == 0:
        raise ValueError("No features available")

    # 🔒 CRITICAL: enforce time order
    X, y = ensure_time_order(X, y)

    # =========================
    # SPLIT
    # =========================
    if problem_type == "regression":
        split_idx = int(len(X) * 0.8)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # =========================
    # CV STRATEGY
    # =========================
    if problem_type == "classification":
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "f1_weighted"
    else:
        cv      = TimeSeriesSplit(n_splits=5)
        scoring = "r2"

    # =========================
    # MODELS
    # =========================
    if problem_type == "classification":
        models = {
            "Random Forest":    RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting":GradientBoostingClassifier(),
            "XGBoost":          XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42),
            "LightGBM":         LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
            "CatBoost":         CatBoostClassifier(iterations=200, verbose=False, random_state=42)
        }
    else:
        models = {
            "Random Forest":    RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting":GradientBoostingRegressor(),
            "XGBoost":          XGBRegressor(n_estimators=200, random_state=42),
            "LightGBM":         LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
            "CatBoost":         CatBoostRegressor(iterations=200, verbose=False, random_state=42)
        }

    # =========================
    # TRAIN LOOP
    # =========================
    results      = []
    best_score   = -np.inf
    best_model   = None
    best_name    = None
    best_metrics = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # -------------------------
            # HOLDOUT
            # -------------------------
            if problem_type == "regression":
                r2   = r2_score(y_test, preds)
                mae  = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                holdout = {"r2": r2, "mae": mae, "rmse": rmse}
                score   = r2

            else:
                acc = accuracy_score(y_test, preds)
                f1  = f1_score(y_test, preds, average="weighted", zero_division=0)

                holdout = {"accuracy": acc, "f1": f1}
                score   = f1

            # -------------------------
            # CV (TRAIN ONLY, TIME SAFE)
            # -------------------------
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring
            )

            cv_metrics = {
                "mean": float(cv_scores.mean()),
                "std":  float(cv_scores.std())
            }

            results.append({
                "Model":   name,
                "Score":   round(score, 4),
                "CV Mean": round(cv_metrics["mean"], 4),
                "CV Std":  round(cv_metrics["std"], 4),
            })

            if score > best_score:
                best_score   = score
                best_model   = model
                best_name    = name
                best_metrics = {
                    "holdout": holdout,
                    "cv": cv_metrics
                }

        except Exception:
            continue

    if best_model is None:
        raise ValueError("All models failed")

    # =========================
    # SAVE
    # =========================
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(X.columns.tolist(), "models/features.pkl")
    joblib.dump(best_metrics, "models/metrics.pkl")

    return pd.DataFrame(results), best_name, best_metrics
