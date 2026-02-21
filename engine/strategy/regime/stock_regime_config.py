"""
Configuration for rule-based stock regime detection.

All windows, thresholds, and hysteresis parameters are centralized here.
Thresholds are calibrated for US equities with daily bars.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StockRegimeConfig:
    """
    Configuration for RuleBasedStockRegimeDetector.

    Parameters are grouped by function: feature windows, stress detection,
    trend detection, range detection, and stability controls.

    Attributes
    ----------
    trend_window : int
        Window for normalized-slope regression on log(close).
    ma_long : int
        Long moving-average period for secondary trend signal.
    rv_lookback : int
        Lookback for rv20 z-score history (default 504 = 2 years).
    rv_lookback_min : int
        Minimum history required to compute rv20 z-score.
    dd_window : int
        Window for rolling max-drawdown calculation.
    annualization : float
        sqrt(252), used if we ever re-derive rv20 locally.

    enter_stress_z : float
        Vol z-score threshold to enter STRESSED_HIGH_VOL.
    exit_stress_z : float
        Vol z-score threshold to exit STRESSED_HIGH_VOL (hysteresis).
    enter_stress_dd : float
        Drawdown threshold to enter STRESSED_HIGH_VOL (negative, e.g. -0.10).
    exit_stress_dd : float
        Drawdown threshold to exit STRESSED_HIGH_VOL (hysteresis, less negative).

    enter_trend_slope : float
        Normalized slope threshold to enter a TRENDING regime.
    exit_trend_slope : float
        Normalized slope threshold to exit a TRENDING regime (hysteresis).
    range_slope_ceil : float
        Max abs(normalized_slope) to qualify as RANGE_LOW_VOL.
    low_vol_pct : float
        Rv20 percentile threshold (vs history) below which stock is considered
        low-vol. Used as annualized rv20 ceiling for RANGE_LOW_VOL.

    confidence_gate : float
        Minimum confidence to allow a regime switch.
    min_regime_duration : int
        Days a regime must persist before another switch is allowed.
        Exception: always allow immediate entry into STRESSED_HIGH_VOL.

    market_stress_z_adj : float
        Amount to subtract from enter_stress_z when market is stressed,
        making it easier for stocks to enter STRESSED_HIGH_VOL.
    market_stress_conf_penalty : float
        Confidence penalty multiplier when market is stressed (0..1).
        Applied to non-stress regime confidence to discourage trend calls
        during market-wide drawdowns.
    """

    # ── Feature windows ─────────────────────────────────────────────────
    trend_window: int = 50
    ma_long: int = 200
    rv_lookback: int = 504
    rv_lookback_min: int = 60
    dd_window: int = 90
    annualization: float = 15.874507866  # sqrt(252)

    # ── Stress thresholds (hysteresis: enter harder, exit easier) ───────
    # Drawdown thresholds are calibrated for individual equities:
    # -20% over 90 days is genuine stress; -10% is too common in normal
    # markets for stocks with 25-50% annualised volatility.
    enter_stress_z: float = 1.5
    exit_stress_z: float = 1.0
    enter_stress_dd: float = -0.20
    exit_stress_dd: float = -0.12

    # ── Trend thresholds (hysteresis) ──────────────────────────────────
    enter_trend_slope: float = 0.5
    exit_trend_slope: float = 0.2

    # ── Range / low-vol thresholds ─────────────────────────────────────
    range_slope_ceil: float = 0.3
    low_vol_pct: float = 0.25  # annualized rv20 ceiling

    # ── Stability controls ─────────────────────────────────────────────
    confidence_gate: float = 0.6
    min_regime_duration: int = 7

    # ── Market-regime conditioning ─────────────────────────────────────
    market_stress_z_adj: float = 0.3
    market_stress_conf_penalty: float = 0.8

    def __post_init__(self) -> None:
        if self.exit_stress_z > self.enter_stress_z:
            raise ValueError(
                "exit_stress_z must be <= enter_stress_z for hysteresis"
            )
        if self.exit_stress_dd < self.enter_stress_dd:
            raise ValueError(
                "exit_stress_dd must be >= enter_stress_dd for hysteresis"
            )
        if self.exit_trend_slope > self.enter_trend_slope:
            raise ValueError(
                "exit_trend_slope must be <= enter_trend_slope for hysteresis"
            )
        if not 0 <= self.confidence_gate <= 1:
            raise ValueError("confidence_gate must be in [0, 1]")
        if self.min_regime_duration < 1:
            raise ValueError("min_regime_duration must be >= 1")
