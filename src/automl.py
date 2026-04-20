import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, TimeSeriesSplit
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


def detect_problem_type(y):
    return "classification" if y.nunique() <= 10 else "regression"


def train_models(X, y, problem_type):

    if len(X) < 30:
        raise ValueError("Dataset too small")

    if problem_type == "regression":
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # CV
    if problem_type == "regression":
        cv = TimeSeriesSplit(n_splits=5)
    else:
        cv = 5

    # Models
    if problem_type == "regression":
        models = {
            "XGBoost": XGBRegressor(n_estimators=200),
            "LightGBM": LGBMRegressor(n_estimators=200),
            "CatBoost": CatBoostRegressor(iterations=200, verbose=False),
        }
    else:
        models = {
            "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(n_estimators=200),
            "CatBoost": CatBoostClassifier(iterations=200, verbose=False),
        }

    best_score = -np.inf
    best_model = None
    best_name = None
    best_metrics = {}
    results = []

    for name, model in models.items():

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # HOLDOUT
            if problem_type == "regression":
                score = r2_score(y_test, preds)
                holdout = {
                    "r2": score,
                    "mae": mean_absolute_error(y_test, preds),
                    "rmse": np.sqrt(mean_squared_error(y_test, preds))
                }
            else:
                score = f1_score(y_test, preds, average="weighted")
                holdout = {
                    "accuracy": accuracy_score(y_test, preds),
                    "f1": score
                }

            # MANUAL CV (SAFE)
            cv_scores = []

            if problem_type == "regression":
                for tr_idx, val_idx in cv.split(X_train):
                    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_val)
                    cv_scores.append(r2_score(y_val, pred))

            else:
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            results.append({
                "Model": name,
                "Score": round(score, 4),
                "CV Mean": round(cv_mean, 4),
                "CV Std": round(cv_std, 4)
            })

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
                best_metrics = {
                    "holdout": holdout,
                    "cv": {"mean": cv_mean, "std": cv_std}
                }

        except:
            continue

    if best_model is None:
        raise ValueError("All models failed")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    return pd.DataFrame(results), best_name, best_metrics
