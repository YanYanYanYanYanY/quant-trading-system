"""
Regime-aware long-only alpha strategy.

Concrete :class:`Strategy` subclass that orchestrates the full
signal-to-order pipeline each bar:

1. Feature update
2. Market regime detection
3. Stock regime classification
4. Alpha scoring (long-only ensemble)
5. Portfolio construction (rank-weighted)
6. Optional risk overlays (strategy-level, account-level)
7. Execution planning (diff to orders)

All heavy components are **injected** at construction time so the
strategy itself is a thin coordinator with no model or data logic.
This keeps it testable with mocks / stubs and lets the backtest or
live runner swap implementations freely.

Bar contract
------------
The bar object must expose at minimum:

* ``bar["asof"]`` or ``bar.asof`` — ``pd.Timestamp``
* ``bar["symbol"]`` or ``bar.symbol`` — ``str`` (optional, for
  single-symbol bars)
* ``bar["close"]`` or ``bar.close`` — ``float``

The strategy accesses ``asof`` via :func:`_get_asof` which tries
attribute access first, then dict lookup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import Order, Strategy
from .portfolio.types import (
    PortfolioConstraints,
    TargetPortfolio,
    normalize_positive,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bar accessor helper
# ---------------------------------------------------------------------------


def _get_asof(bar: Any) -> pd.Timestamp:
    """
    Extract the ``asof`` timestamp from a bar.

    Tries attribute access (``bar.asof``) then dict lookup
    (``bar["asof"]``).  Falls back to ``pd.Timestamp.now()``
    as a last resort so the pipeline never crashes on a missing
    timestamp.
    """
    if hasattr(bar, "asof"):
        return bar.asof
    if isinstance(bar, dict) and "asof" in bar:
        return bar["asof"]
    return pd.Timestamp.now()


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class RegimeAlphaStrategy(Strategy):
    """
    Regime-aware long-only alpha strategy.

    Coordinates feature computation, regime detection, alpha scoring,
    portfolio construction, and optional risk overlays into a single
    bar-driven lifecycle.

    Parameters
    ----------
    warmup_bars : int
        Number of bars the feature engine needs before indicators
        are reliable.
    universe : list[str]
        Tradeable symbol universe.
    portfolio_constraints : PortfolioConstraints
        Position-count, weight, and turnover limits.
    feature_engine : Any
        Must expose ``update(bar)`` and either
        ``get_feature_bundle(asof, universe)`` or a ``.features``
        attribute.
    market_regime_detector : Any
        Must expose ``detect(asof, features, universe)`` or
        ``update(df) -> MarketRegimeState``.
    stock_regime_detector : Any
        Must expose ``detect(asof, market_regime, features, universe)``.
    alpha_engine : Any
        Must expose ``generate(asof, market_regime, stock_regimes,
        features, universe) -> AlphaVector``.
    portfolio_constructor : Any
        Must expose ``build_targets(asof, alpha, universe,
        constraints, current_weights) -> TargetPortfolio``.
    strategy_risk_overlay : Any, optional
        If provided, must expose ``adjust(asof, target_portfolio,
        market_regime, stock_regimes, features) -> TargetPortfolio``.
    account_risk_overlay : Any, optional
        If provided, must expose ``adjust(asof, target_portfolio,
        account_state) -> TargetPortfolio``.
    execution_planner : Any, optional
        If provided, must expose ``diff_to_orders(current_weights,
        target_weights) -> list[Order]``.

    Attributes
    ----------
    current_weights : dict[str, float]
        Last committed portfolio weights.  Updated after each live
        bar; starts empty.

    Examples
    --------
    >>> strategy = RegimeAlphaStrategy(
    ...     warmup_bars=200,
    ...     universe=["AAPL", "MSFT", "GOOG"],
    ...     portfolio_constraints=PortfolioConstraints(),
    ...     feature_engine=feat_eng,
    ...     market_regime_detector=mkt_det,
    ...     stock_regime_detector=stk_det,
    ...     alpha_engine=alpha_eng,
    ...     portfolio_constructor=port_con,
    ... )
    """

    def __init__(
        self,
        warmup_bars: int,
        universe: List[str],
        portfolio_constraints: PortfolioConstraints,
        feature_engine: Any,
        market_regime_detector: Any,
        stock_regime_detector: Any,
        alpha_engine: Any,
        portfolio_constructor: Any,
        strategy_risk_overlay: Any = None,
        account_risk_overlay: Any = None,
        execution_planner: Any = None,
    ) -> None:
        super().__init__(warmup_bars=warmup_bars)

        self.universe = list(universe)
        self.portfolio_constraints = portfolio_constraints

        # Injected components
        self._feature_engine = feature_engine
        self._market_regime_detector = market_regime_detector
        self._stock_regime_detector = stock_regime_detector
        self._alpha_engine = alpha_engine
        self._portfolio_constructor = portfolio_constructor
        self._strategy_risk_overlay = strategy_risk_overlay
        self._account_risk_overlay = account_risk_overlay
        self._execution_planner = execution_planner

        # Portfolio state
        self.current_weights: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle: warmup
    # ------------------------------------------------------------------

    def warmup_bar(self, bar: Any) -> None:
        """
        Feed one bar to the feature engine during warmup.

        Updates the feature engine (and any detector that exposes an
        ``update`` method) but does **not** run regimes, alpha, or
        portfolio construction.

        Parameters
        ----------
        bar : Any
            A single bar of market data.
        """
        # 1. Feature update
        self._feature_engine.update(bar)

        # 2. Let detectors absorb data if they support it
        if hasattr(self._market_regime_detector, "update"):
            self._market_regime_detector.update(bar)
        if hasattr(self._stock_regime_detector, "update"):
            self._stock_regime_detector.update(bar)

    # ------------------------------------------------------------------
    # Lifecycle: live
    # ------------------------------------------------------------------

    def on_bar(self, bar: Any) -> Any:
        """
        Run the full signal-to-order pipeline on a live bar.

        Steps
        -----
        1. Update feature engine.
        2. Extract ``asof`` and ``features``.
        3. Detect market regime.
        4. Classify stock regimes.
        5. Generate alpha vector.
        6. Construct target portfolio.
        7. Apply optional overlays.
        8. Defensive long-only clip.
        9. Diff to orders (if execution planner provided).
        10. Commit weights.

        Parameters
        ----------
        bar : Any
            A single bar of market data.

        Returns
        -------
        list[Order] | TargetPortfolio
            Orders if an execution planner is configured, otherwise
            the final :class:`TargetPortfolio`.
        """
        # ── 1. Feature update ──────────────────────────────────────────
        self._feature_engine.update(bar)

        # ── 2. Timestamp & features ────────────────────────────────────
        asof = _get_asof(bar)
        features = self._get_features(asof)

        # ── 3. Market regime ───────────────────────────────────────────
        market_regime = self._detect_market_regime(asof, features)

        # ── 4. Stock regimes ───────────────────────────────────────────
        stock_regimes = self._stock_regime_detector.detect(
            asof, market_regime, features, self.universe,
        )

        # ── 5. Alpha vector ───────────────────────────────────────────
        alpha_vec = self._alpha_engine.generate(
            asof=asof,
            market_regime=market_regime,
            stock_regimes=stock_regimes,
            features=features,
            universe=self.universe,
        )

        # ── 6. Portfolio construction ──────────────────────────────────
        portfolio = self._portfolio_constructor.build_targets(
            asof=asof,
            alpha=alpha_vec,
            universe=self.universe,
            constraints=self.portfolio_constraints,
            current_weights=self.current_weights or None,
        )

        # ── 7. Overlays ───────────────────────────────────────────────
        if self._strategy_risk_overlay is not None:
            portfolio = self._strategy_risk_overlay.adjust(
                asof, portfolio, market_regime, stock_regimes, features,
            )

        if self._account_risk_overlay is not None:
            portfolio = self._account_risk_overlay.adjust(
                asof, portfolio, None,  # account_state not yet wired
            )

        # ── 8. Defensive long-only clip ────────────────────────────────
        final_weights = _clip_long_only(portfolio.weights)

        # ── 9. Orders / weight diff ────────────────────────────────────
        if self._execution_planner is not None:
            result = self._execution_planner.diff_to_orders(
                self.current_weights, final_weights,
            )
        else:
            result = TargetPortfolio(
                asof=asof,
                weights=final_weights,
                gross=sum(final_weights.values()),
                reasons=portfolio.reasons,
            )

        # ── 10. Commit weights ─────────────────────────────────────────
        self.current_weights = dict(final_weights)

        # ── Lightweight logging ────────────────────────────────────────
        self._log_bar_summary(asof, final_weights, market_regime)

        return result

    # ------------------------------------------------------------------
    # Date-driven interface (multi-symbol runner)
    # ------------------------------------------------------------------

    def process_date(self, asof: pd.Timestamp) -> Any:
        """Run the full cross-sectional pipeline for a date.

        Unlike :meth:`on_bar`, this method does **not** call
        ``feature_engine.update(bar)`` — the caller is responsible for
        feeding all symbol bars to the feature engine before invoking
        this method.  This decoupling is essential for multi-symbol
        backtests where the runner feeds N bars per date and then
        triggers the pipeline once.

        Steps 2-10 are identical to :meth:`on_bar`.

        Parameters
        ----------
        asof : pd.Timestamp
            Current evaluation date.

        Returns
        -------
        list[Order] | TargetPortfolio
            Same semantics as :meth:`on_bar`.
        """
        # ── 2. Features ───────────────────────────────────────────────
        features = self._get_features(asof)

        # ── 3. Market regime ───────────────────────────────────────────
        market_regime = self._detect_market_regime(asof, features)

        # ── 4. Stock regimes ───────────────────────────────────────────
        stock_regimes = self._stock_regime_detector.detect(
            asof, market_regime, features, self.universe,
        )

        # ── 5. Alpha vector ───────────────────────────────────────────
        alpha_vec = self._alpha_engine.generate(
            asof=asof,
            market_regime=market_regime,
            stock_regimes=stock_regimes,
            features=features,
            universe=self.universe,
        )

        # ── 6. Portfolio construction ──────────────────────────────────
        portfolio = self._portfolio_constructor.build_targets(
            asof=asof,
            alpha=alpha_vec,
            universe=self.universe,
            constraints=self.portfolio_constraints,
            current_weights=self.current_weights or None,
        )

        # ── 7. Overlays ───────────────────────────────────────────────
        if self._strategy_risk_overlay is not None:
            portfolio = self._strategy_risk_overlay.adjust(
                asof, portfolio, market_regime, stock_regimes, features,
            )

        if self._account_risk_overlay is not None:
            portfolio = self._account_risk_overlay.adjust(
                asof, portfolio, None,
            )

        # ── 8. Defensive long-only clip ────────────────────────────────
        final_weights = _clip_long_only(portfolio.weights)

        # ── 9. Orders / weight diff ────────────────────────────────────
        if self._execution_planner is not None:
            result = self._execution_planner.diff_to_orders(
                self.current_weights, final_weights,
            )
        else:
            result = TargetPortfolio(
                asof=asof,
                weights=final_weights,
                gross=sum(final_weights.values()),
                reasons=portfolio.reasons,
            )

        # ── 10. Commit weights ─────────────────────────────────────────
        self.current_weights = dict(final_weights)

        # ── Lightweight logging ────────────────────────────────────────
        self._log_bar_summary(asof, final_weights, market_regime)

        return result

    # ------------------------------------------------------------------
    # Reset (extend base)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear lifecycle state and committed weights."""
        super().reset()
        self.current_weights = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_features(self, asof: pd.Timestamp) -> Any:
        """
        Retrieve the current feature bundle from the feature engine.

        Tries ``get_feature_bundle(asof, universe)`` first, then
        falls back to the ``.features`` attribute.
        """
        if hasattr(self._feature_engine, "get_feature_bundle"):
            return self._feature_engine.get_feature_bundle(
                asof, self.universe,
            )
        return self._feature_engine.features

    def _detect_market_regime(self, asof: pd.Timestamp, features: Any) -> Any:
        """
        Detect market regime using whatever interface the detector
        exposes.

        Tries ``detect(asof, features, universe)`` first, then
        ``update(features)`` for detectors that accept a DataFrame.
        """
        det = self._market_regime_detector

        if hasattr(det, "detect"):
            return det.detect(asof, features, self.universe)

        # Fallback: the SPY MarketRegimeDetector uses update(df)
        if hasattr(det, "update"):
            return det.update(features)

        raise TypeError(
            f"{type(det).__name__} has neither detect() nor update()"
        )

    @staticmethod
    def _log_bar_summary(
        asof: pd.Timestamp,
        weights: Dict[str, float],
        market_regime: Any,
    ) -> None:
        """Emit a DEBUG-level summary of the bar result."""
        if not log.isEnabledFor(logging.DEBUG):
            return

        n_pos = sum(1 for w in weights.values() if w > 0)
        gross = sum(weights.values())

        # Top 3 weights, sorted descending then alphabetical
        top = sorted(
            ((s, w) for s, w in weights.items() if w > 0),
            key=lambda kv: (-kv[1], kv[0]),
        )[:3]
        top_str = ", ".join(f"{s}={w:.3f}" for s, w in top)

        regime_label = (
            market_regime.label.value
            if hasattr(market_regime, "label")
            else str(market_regime)
        )

        log.debug(
            "bar %s | regime=%s | n_pos=%d | gross=%.4f | top=[%s]",
            asof.date(), regime_label, n_pos, gross, top_str,
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _clip_long_only(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Defensively clip any negative weights to 0 and renormalise.

    In a correctly functioning pipeline every weight should already
    be ``>= 0``.  This guard catches bugs in overlays or custom
    constructors without crashing.

    Parameters
    ----------
    weights : dict[str, float]
        Proposed weights (may contain negatives due to bugs).

    Returns
    -------
    dict[str, float]
        Guaranteed non-negative weights.  If any clipping occurred
        the remaining positive weights are renormalised to preserve
        their original sum (but capped at the pre-clip total).
    """
    if not weights:
        return weights

    has_negative = any(w < 0.0 for w in weights.values())
    if not has_negative:
        return weights

    log.warning(
        "Negative weights detected (%d symbols) — clipping to 0",
        sum(1 for w in weights.values() if w < 0),
    )

    original_gross = sum(w for w in weights.values() if w > 0)
    clipped = {s: max(w, 0.0) for s, w in weights.items()}
    clipped = {s: w for s, w in clipped.items() if w > 0}

    if not clipped:
        return {}

    if original_gross > 0:
        clipped = normalize_positive(clipped, target_sum=original_gross)

    return clipped
