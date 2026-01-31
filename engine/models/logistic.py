from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseModel, ModelMeta

Penalty = Literal["l1", "l2", "elasticnet"]
class LogisticRegressionModel(BaseModel):
    """
    Binary classifier: predicts P(y=1) using a scaler + logistic regression.
    """

    def __init__(
        self,
        *,
        feature_cols: list[str],
        C: float = 1.0,
        penalty: Penalty = "l2",
        max_iter: int = 2000,
        n_jobs: int = -1,
        class_weight: Optional[Dict[int, float] | str] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        self.feature_cols = feature_cols

        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=C,
                    penalty=penalty,
                    max_iter=max_iter,
                    n_jobs=n_jobs,
                    class_weight=class_weight,
                    random_state=random_state,
                    solver="lbfgs" if penalty == "l2" else "liblinear",
                )),
            ]
        )

        created_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self.meta = ModelMeta(
            model_name="logistic_regression",
            feature_cols=feature_cols,
            created_utc=created_utc,
            extra={
                "C": C,
                "penalty": penalty,
                "max_iter": max_iter,
                "class_weight": class_weight,
                "random_state": random_state,
            },
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_use = X[self.feature_cols].copy()
        y_use = y.astype("int64")
        self.pipeline.fit(X_use, y_use)

    def predict_proba_1d(self, X: pd.DataFrame) -> np.ndarray:
        X_use = X[self.feature_cols].copy()
        proba_2d = self.pipeline.predict_proba(X_use)
        return proba_2d[:, 1]

    def save(self, path: str) -> None:
        """
        Saves a dict containing:
          - pipeline
          - meta
        """
        payload = {
            "pipeline": self.pipeline,
            "meta": asdict(self.meta) if self.meta else None,
            "feature_cols": self.feature_cols,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        payload = joblib.load(path)

        feature_cols = payload["feature_cols"]
        obj = cls(feature_cols=feature_cols)  # create default instance
        obj.pipeline = payload["pipeline"]

        meta = payload.get("meta")
        if meta is not None:
            obj.meta = ModelMeta(**meta)
        else:
            obj.meta = None

        return obj