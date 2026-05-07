"""Train and evaluate shot-danger models.

We treat shot danger as goal probability: P(goal | features). The baseline is
logistic regression because (a) it gives interpretable coefficients a coach
can reason about, and (b) it produces calibrated probabilities out of the box.

A gradient-boosted variant is included for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    """Bundle the trained model with the slice of data used to evaluate it."""
    model: object
    feature_names: list[str]
    metrics: dict = field(default_factory=dict)
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None
    y_pred_proba: np.ndarray | None = None


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = "goal",
    test_size: float = 0.2,
    random_state: int = 7,
):
    if not feature_cols:
        raise ValueError("No usable model features were found after filtering.")

    X = df[feature_cols].astype(float)
    y = df[target].astype(int)
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "Need both goals and non-goals to train a goal-probability model. "
            "Try broadening the filters or using more data."
        )
    if class_counts.min() < 2:
        raise ValueError(
            "Need at least two examples of each class for a stratified train/test split. "
            f"Class counts: {class_counts.to_dict()}"
        )
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )


def train_logistic(X_train, y_train) -> Pipeline:
    """Logistic regression with feature scaling.

    We do *not* use class_weight='balanced' here. Goals are ~7% of shots, so
    balancing inflates predicted probabilities and breaks calibration —
    fatal for an xG-style model where a 0.05 prediction should mean 5%
    long-run goal rate. The unweighted model is well-calibrated and still
    ranks shots correctly (which is what AUC measures).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=25,
        n_jobs=-1,
        random_state=7,
    )
    rf.fit(X_train, y_train)
    return rf


def train_gradient_boosting(X_train, y_train) -> GradientBoostingClassifier:
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=7,
    )
    gb.fit(X_train, y_train)
    return gb


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Standard probability metrics for an imbalanced goal model.

    The confusion matrix is returned only as an optional sanity check. A 0.5
    threshold is not a meaningful main metric for xG-style probabilities.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = probability_metrics(y_test, proba)
    metrics.update({
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "actual_goal_rate": float(y_test.mean()),
        "predicted_goal_rate": float(proba.mean()),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    })
    return metrics


def probability_metrics(y_true, y_pred_proba) -> dict:
    """Proper scoring rules for probability predictions.

    Accuracy is intentionally absent: goals are rare, so a model can be very
    accurate by predicting "no goal" for almost every shot while being useless
    as an xG model.
    """
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(y_pred_proba).astype(float)
    proba = np.clip(proba, 1e-15, 1 - 1e-15)
    return {
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y, proba)),
    }


def benchmark_probabilities(
    y_true,
    model_proba,
    benchmark_proba,
    benchmark_name: str = "benchmark",
) -> pd.DataFrame:
    """Compare our probabilities with an external xG benchmark."""
    rows = []
    for name, proba in [("our_model", model_proba), (benchmark_name, benchmark_proba)]:
        base = probability_metrics(y_true, proba)
        base["roc_auc"] = float(roc_auc_score(y_true, proba))
        base["mean_predicted"] = float(np.mean(proba))
        base["model"] = name
        rows.append(base)
    return (
        pd.DataFrame(rows)
        [["model", "roc_auc", "log_loss", "brier", "mean_predicted"]]
        .reset_index(drop=True)
    )


def calibration_table(y_true, y_pred_proba, n_bins: int = 10) -> pd.DataFrame:
    """Decile bins of predicted probability vs. observed goal rate.

    A well-calibrated model has mean_pred ~= actual within each bin.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred_proba)})
    if df["p"].nunique() <= 1:
        df["bin"] = 0
    else:
        df["bin"] = pd.qcut(df["p"], q=min(n_bins, len(df)), duplicates="drop")
    return (
        df.groupby("bin", observed=True)
          .agg(n=("y", "size"), mean_pred=("p", "mean"), actual=("y", "mean"))
          .reset_index(drop=True)
    )


def coefficient_table(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Logistic regression coefficients in the standardized feature space.

    A coefficient of +0.4 means: a one-standard-deviation increase in that
    feature multiplies the log-odds of a goal by ~0.4.
    """
    clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
    coefs = clf.coef_[0]
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
          .assign(abs_coef=lambda d: d["coefficient"].abs())
          .sort_values("abs_coef", ascending=False)
          .drop(columns="abs_coef")
          .reset_index(drop=True)
    )


def feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    """Return importances for tree models; coefficients otherwise."""
    target = model
    if hasattr(model, "named_steps"):
        target = model.named_steps.get("clf", model)
    if hasattr(target, "feature_importances_"):
        return (
            pd.DataFrame({
                "feature": feature_names,
                "importance": target.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return coefficient_table(model, feature_names)
