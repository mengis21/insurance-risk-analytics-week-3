"""Task 4 modelling workflow for risk-adjusted pricing."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if not hasattr(np, "obj2sctype"):
    def _obj2sctype(dtype):  # type: ignore[override]
        return np.dtype(dtype).type

    np.obj2sctype = _obj2sctype  # type: ignore[attr-defined]

import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import load_enriched_dataset

from lightgbm import LGBMClassifier, LGBMRegressor

RANDOM_STATE = 42
REPORTS_DIR = Path("reports")
IMAGES_DIR = REPORTS_DIR / "images"
METRICS_JSON = REPORTS_DIR / "task4_model_metrics.json"
RISK_PREMIUM_CSV = REPORTS_DIR / "risk_premium_segments.csv"
PREDICTIONS_CSV = REPORTS_DIR / "model_predictions_sample.csv"
SHAP_FIG_PATH = IMAGES_DIR / "severity_feature_importance.png"
MARGIN_FACTOR = 1.10  # 10% profit loading over expected claim cost


def _build_preprocessor() -> ColumnTransformer:
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["Smoker", "Gender", "Province", "PostalCode"]
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("numerical", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def _train_regression_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Pipeline], Dict[str, np.ndarray]]:
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "lightgbm": LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1),
    }
    metrics: Dict[str, Dict[str, float]] = {}
    trained: Dict[str, Pipeline] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics[name] = _regression_metrics(y_test, preds)
        trained[name] = pipeline
        predictions[name] = preds

    return metrics, trained, predictions


def _train_classification_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Pipeline], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, class_weight="balanced"),
        "lightgbm": LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE, class_weight="balanced"),
    }
    metrics: Dict[str, Dict[str, float]] = {}
    trained: Dict[str, Pipeline] = {}
    predictions: Dict[str, np.ndarray] = {}
    probabilities: Dict[str, np.ndarray] = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics[name] = _classification_metrics(y_test, preds, proba)
        trained[name] = pipeline
        predictions[name] = preds
        probabilities[name] = proba

    return metrics, trained, predictions, probabilities


def _clean_for_json(value):
    if isinstance(value, dict):
        return {key: _clean_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_for_json(item) for item in value]
    if isinstance(value, (np.floating, float, np.integer, int)):
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _generate_shap_plot(model: Pipeline, X_train: pd.DataFrame) -> None:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    if X_train.shape[0] > 500:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_indices = rng.choice(X_train.index, size=500, replace=False)
        X_sample = X_train.loc[sample_indices]
    else:
        X_sample = X_train
    transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
    except Exception:
        explainer = shap.LinearExplainer(estimator, transformed)
        shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_FIG_PATH, dpi=300)
    plt.close()


def _select_best_classifier(metrics: Dict[str, Dict[str, float]]) -> str:
    best_name = None
    best_score = -float("inf")
    for name, vals in metrics.items():
        score = vals.get("roc_auc")
        if score is None or (isinstance(score, float) and math.isnan(score)):
            score = vals.get("f1", -float("inf"))
        if score > best_score:
            best_score = score
            best_name = name
    if best_name is None:
        raise ValueError("Unable to select a best classifier; all scores undefined")
    return best_name


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df, _ = load_enriched_dataset()
    feature_cols = ["age", "bmi", "children", "Smoker", "Gender", "Province", "PostalCode"]
    dataset = df.dropna(subset=feature_cols + ["TotalClaims", "TotalPremium", "HighSeverityFlag"])

    X = dataset[feature_cols]
    y_severity = dataset["TotalClaims"]
    y_premium = dataset["TotalPremium"]
    y_flag = dataset["HighSeverityFlag"].astype(int)

    (
        X_train,
        X_test,
        y_sev_train,
        y_sev_test,
        y_prem_train,
        y_prem_test,
        y_flag_train,
        y_flag_test,
    ) = train_test_split(
        X,
        y_severity,
        y_premium,
        y_flag,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_flag,
    )

    severity_metrics, severity_models, severity_preds = _train_regression_models(X_train, X_test, y_sev_train, y_sev_test)
    premium_metrics, premium_models, premium_preds = _train_regression_models(X_train, X_test, y_prem_train, y_prem_test)
    class_metrics, class_models, class_preds, class_prob = _train_classification_models(
        X_train, X_test, y_flag_train, y_flag_test
    )

    best_severity_name = min(severity_metrics, key=lambda name: severity_metrics[name]["rmse"])
    best_premium_name = min(premium_metrics, key=lambda name: premium_metrics[name]["rmse"])
    best_class_name = _select_best_classifier(class_metrics)

    best_severity_model = severity_models[best_severity_name]
    best_class_model = class_models[best_class_name]

    severity_pred = best_severity_model.predict(X_test)
    class_probabilities = class_prob[best_class_name]

    risk_frame = X_test.copy()
    risk_frame["ActualTotalClaims"] = y_sev_test.values
    risk_frame["ActualTotalPremium"] = y_prem_test.values
    risk_frame["PredictedSeverity"] = severity_pred
    risk_frame["HighSeverityProbability"] = class_probabilities
    risk_frame["ExpectedClaim"] = risk_frame["PredictedSeverity"] * risk_frame["HighSeverityProbability"]
    risk_frame["RiskBasedPremium"] = risk_frame["ExpectedClaim"] * MARGIN_FACTOR
    risk_frame["PremiumDelta"] = risk_frame["RiskBasedPremium"] - risk_frame["ActualTotalPremium"]

    segment_summary = (
        risk_frame
        .groupby(["Province", "PostalCode"])
        .agg(
            policies=("RiskBasedPremium", "size"),
            avg_actual_premium=("ActualTotalPremium", "mean"),
            avg_risk_premium=("RiskBasedPremium", "mean"),
            avg_margin_delta=("PremiumDelta", "mean"),
            avg_high_severity_prob=("HighSeverityProbability", "mean"),
        )
        .reset_index()
    )
    segment_summary.to_csv(RISK_PREMIUM_CSV, index=False)

    sample_predictions = risk_frame.copy()
    sample_predictions["AbsPremiumDelta"] = sample_predictions["PremiumDelta"].abs()
    sample_predictions = sample_predictions.sort_values("AbsPremiumDelta", ascending=False).head(25)
    sample_predictions.drop(columns=["AbsPremiumDelta"], inplace=True)
    sample_predictions.to_csv(PREDICTIONS_CSV, index=False)

    try:
        _generate_shap_plot(best_severity_model, X_train)
    except Exception as exc:  # pragma: no cover - defensive hook
        print(f"SHAP plot generation failed: {exc}")

    metrics_payload = {
        "severity": severity_metrics,
        "premium": premium_metrics,
        "classification": class_metrics,
        "best_models": {
            "severity": best_severity_name,
            "premium": best_premium_name,
            "classification": best_class_name,
        },
    }

    cleaned_metrics = _clean_for_json(metrics_payload)
    with METRICS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(cleaned_metrics, fh, indent=2)

    print("Model training complete.")
    print(f"Best severity model: {best_severity_name}")
    print(f"Best premium model: {best_premium_name}")
    print(f"Best classification model: {best_class_name}")


if __name__ == "__main__":
    main()
