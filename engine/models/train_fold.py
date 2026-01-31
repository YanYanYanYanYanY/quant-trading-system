from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from .base import BaseModel


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    accuracy: float
    auc: float
    logloss: float


def train_one_fold(
    model: BaseModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = 0.5,
) -> FoldResult:
    """
    Train the model on one fold and compute basic classification metrics.
    """
    model.fit(X_train, y_train)

    proba = model.predict_proba_1d(X_test)
    pred = (proba >= threshold).astype(np.int64)

    # Metrics
    acc = float(accuracy_score(y_test, pred))

    # AUC requires both classes present in y_test; handle edge case gracefully
    try:
        auc = float(roc_auc_score(y_test, proba))
    except ValueError:
        auc = float("nan")

    # log_loss also can fail if probabilities are weird; clamp for stability
    eps = 1e-12
    proba_clip = np.clip(proba, eps, 1 - eps)
    ll = float(log_loss(y_test.astype(int), np.c_[1 - proba_clip, proba_clip]))

    return FoldResult(
        fold=-1,  # filled by train_walk_forward
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        accuracy=acc,
        auc=auc,
        logloss=ll,
    )


def train_walk_forward(
    model_factory: Callable[[], BaseModel],
    folds: Iterator[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
    *,
    threshold: float = 0.5,
    max_folds: Optional[int] = None,
) -> pd.DataFrame:
    """
    Train/evaluate across walk-forward folds.

    Parameters
    ----------
    model_factory: callable returning a *fresh* model each fold
    folds: iterator yielding (X_train, y_train, X_test, y_test)
    threshold: classification threshold
    max_folds: optionally stop early

    Returns
    -------
    DataFrame with one row per fold.
    """
    results: list[dict] = []

    for i, (X_train, y_train, X_test, y_test) in enumerate(folds, start=1):
        if max_folds is not None and i > max_folds:
            break

        model = model_factory()
        fr = train_one_fold(
            model,
            X_train, y_train,
            X_test, y_test,
            threshold=threshold,
        )
        fr.fold = i  # type: ignore[assignment]

        results.append({
            "fold": fr.fold,
            "n_train": fr.n_train,
            "n_test": fr.n_test,
            "accuracy": fr.accuracy,
            "auc": fr.auc,
            "logloss": fr.logloss,
        })

    return pd.DataFrame(results)