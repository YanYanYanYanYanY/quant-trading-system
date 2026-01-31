from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelMeta:
    """Lightweight metadata saved alongside the trained model."""
    model_name: str
    feature_cols: list[str]
    created_utc: str  # ISO string, e.g. "2026-01-05T18:22:00Z"
    extra: Dict[str, Any] | None = None


class BaseModel(ABC):
    """
    Contract for models in this codebase.

    Models should:
      - fit on (X, y)
      - output probability of positive class via predict_proba_1d
      - be save/load-able
    """

    meta: Optional[ModelMeta] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    @abstractmethod
    def predict_proba_1d(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability for class 1 (positive class) as shape (n_samples,).
        """
        ...

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba_1d(X)
        return (proba >= threshold).astype(np.int64)

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        ...

    def meta_dict(self) -> dict:
        return asdict(self.meta) if self.meta else {}