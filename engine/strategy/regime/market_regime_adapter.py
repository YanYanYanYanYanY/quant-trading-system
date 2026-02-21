"""
Event-driven adapter for the batch-oriented MarketRegimeDetector.

The :class:`MarketRegimeDetector` expects a full SPY DataFrame via
``update(df)``.  In an event-driven (bar-by-bar) pipeline the adapter
below accumulates SPY bars incrementally and delegates to the inner
detector on each :meth:`detect` call.

Usage
-----
::

    adapter = EventDrivenMarketRegime(spy_symbol="SPY")
    for bar in bar_stream:
        adapter.update(bar)             # accumulate SPY data
    state = adapter.detect(asof, features, universe)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import RegimeConfig
from .market_regime import MarketRegimeDetector
from .market_types import MarketRegimeLabel, MarketRegimeState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bar accessor (mirrors feature_engine._get)
# ---------------------------------------------------------------------------

def _get(bar: Any, field: str, default: Any = None) -> Any:
    if hasattr(bar, field):
        return getattr(bar, field)
    if isinstance(bar, dict):
        return bar.get(field, default)
    return default


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class EventDrivenMarketRegime:
    """Wraps :class:`MarketRegimeDetector` for event-driven use.

    Parameters
    ----------
    spy_symbol : str
        Symbol treated as the market proxy (default ``"SPY"``).
    config : RegimeConfig, optional
        Forwarded to the inner :class:`MarketRegimeDetector`.
    inner : MarketRegimeDetector, optional
        Inject a pre-built detector.  If ``None``, a fresh one is
        created from *config*.

    Attributes
    ----------
    last_state : MarketRegimeState or None
        Most recent regime state returned by :meth:`detect`.
    """

    def __init__(
        self,
        spy_symbol: str = "SPY",
        config: RegimeConfig | None = None,
        inner: MarketRegimeDetector | None = None,
    ) -> None:
        self._spy_symbol = spy_symbol
        self._inner = inner or MarketRegimeDetector(config)
        self._config = self._inner.config

        # Accumulated SPY data — list of (timestamp, close) tuples
        self._spy_records: List[Dict[str, Any]] = []
        self._spy_df: Optional[pd.DataFrame] = None  # cached DataFrame
        self._dirty = True  # True when _spy_records has grown since last _build_df

        self.last_state: Optional[MarketRegimeState] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def spy_symbol(self) -> str:
        return self._spy_symbol

    @property
    def n_spy_bars(self) -> int:
        """Number of SPY bars accumulated so far."""
        return len(self._spy_records)

    @property
    def min_bars_needed(self) -> int:
        """Minimum SPY bars before regime detection can run."""
        return self._config.rv_window + 10

    # ------------------------------------------------------------------
    # update — accumulate bars
    # ------------------------------------------------------------------

    def update(self, bar: Any) -> None:
        """Ingest a bar.  If it belongs to the SPY symbol, accumulate it.

        Parameters
        ----------
        bar
            Must expose ``symbol``, ``close``, ``asof`` (via attribute
            or dict access).  Non-SPY bars are silently ignored.
        """
        sym = _get(bar, "symbol")
        if sym != self._spy_symbol:
            return

        close = float(_get(bar, "close"))
        asof = _get(bar, "asof")

        self._spy_records.append({"close": close, "timestamp": asof})
        self._dirty = True

    # ------------------------------------------------------------------
    # detect — run the inner detector
    # ------------------------------------------------------------------

    def detect(
        self,
        asof: pd.Timestamp,
        features: Any = None,
        universe: List[str] | None = None,
    ) -> MarketRegimeState:
        """Detect market regime from accumulated SPY data.

        Parameters
        ----------
        asof : pd.Timestamp
            Current evaluation timestamp.
        features, universe
            Accepted for interface compatibility with the
            ``RegimeAlphaStrategy._detect_market_regime`` protocol.
            Not used.

        Returns
        -------
        MarketRegimeState
            If not enough SPY data has been accumulated, returns a
            default ``CALM_RANGE`` state with low confidence.
        """
        if len(self._spy_records) < self.min_bars_needed:
            log.debug(
                "Not enough SPY data (%d/%d) — returning default regime",
                len(self._spy_records),
                self.min_bars_needed,
            )
            self.last_state = MarketRegimeState(
                asof=asof,
                label=MarketRegimeLabel.CALM_RANGE,
                confidence=0.0,
                scores={"reason": "insufficient_data"},
            )
            return self.last_state

        # Build / refresh the SPY DataFrame
        if self._dirty or self._spy_df is None:
            self._spy_df = self._build_df()
            self._dirty = False

        self.last_state = self._inner.update(self._spy_df)
        return self.last_state

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated data and inner detector state."""
        self._spy_records.clear()
        self._spy_df = None
        self._dirty = True
        self._inner.reset_state()
        self.last_state = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_df(self) -> pd.DataFrame:
        """Convert accumulated records to a DatetimeIndex DataFrame."""
        df = pd.DataFrame(self._spy_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        # Drop duplicate timestamps (keep last)
        df = df[~df.index.duplicated(keep="last")]
        return df

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        regime = self.last_state.label.value if self.last_state else "?"
        return (
            f"EventDrivenMarketRegime("
            f"spy={self._spy_symbol!r}, "
            f"bars={len(self._spy_records)}, "
            f"regime={regime})"
        )
