"""
Configuration dataclass for market regime detection.

All windows, thresholds, and hysteresis parameters are centralized here
for easy tuning and experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegimeConfig:
    """
    Configuration for MarketRegimeDetector.
    
    Attributes
    ----------
    rv_window : int
        Window for realized volatility calculation (default 20 days).
    rv_annualization_factor : float
        Factor to annualize daily vol (sqrt(252) ≈ 15.87).
    vol_lookback_min : int
        Minimum lookback for vol z-score history (1 year = 252).
    vol_lookback_max : int
        Maximum lookback for vol z-score history (3 years = 756).
    vol_lookback_default : int
        Default lookback for vol z-score (2 years = 504).
    drawdown_window : int
        Window for max drawdown calculation (default 90 days).
    trend_window : int
        Window for trend slope calculation (default 50 days).
    ma_short : int
        Short MA period for crossover signal (default 50).
    ma_long : int
        Long MA period for crossover signal (default 200).
    use_ma_crossover : bool
        Whether to use MA crossover as secondary trend signal.
    
    z_vol_stress_enter : float
        Z-score threshold to enter stressed regime.
    z_vol_stress_exit : float
        Z-score threshold to exit stressed regime (hysteresis).
    drawdown_stress_enter : float
        Drawdown threshold to enter stressed (e.g., -0.10 = -10%).
    drawdown_stress_exit : float
        Drawdown threshold to exit stressed (hysteresis).
    
    trend_up_enter : float
        Normalized slope threshold to enter trending up.
    trend_up_exit : float
        Normalized slope threshold to exit trending up (hysteresis).
    trend_down_enter : float
        Normalized slope threshold to enter trending down.
    trend_down_exit : float
        Normalized slope threshold to exit trending down (hysteresis).
    
    confidence_gate : float
        Minimum confidence required to switch regimes (default 0.6).
    min_regime_duration : int
        Minimum days to stay in a regime before switching (default 7).
    
    confidence_margin_scale : float
        How much distance from threshold contributes to confidence.
    confidence_agreement_weight : float
        Weight given to trend measure agreement in confidence calc.
    """
    
    # Volatility parameters
    rv_window: int = 20
    rv_annualization_factor: float = 15.874507866  # sqrt(252)
    vol_lookback_min: int = 252
    vol_lookback_max: int = 756
    vol_lookback_default: int = 504
    
    # Drawdown parameters
    drawdown_window: int = 90
    
    # Trend parameters
    trend_window: int = 50
    ma_short: int = 50
    ma_long: int = 200
    use_ma_crossover: bool = True
    
    # Stress thresholds (hysteresis: enter > exit)
    z_vol_stress_enter: float = 1.5
    z_vol_stress_exit: float = 1.0
    drawdown_stress_enter: float = -0.10  # -10%
    drawdown_stress_exit: float = -0.07   # -7%
    
    # Trend thresholds (hysteresis)
    trend_up_enter: float = 0.5
    trend_up_exit: float = 0.2
    trend_down_enter: float = -0.5
    trend_down_exit: float = -0.2
    
    # Stability controls
    confidence_gate: float = 0.6
    min_regime_duration: int = 7
    
    # Confidence calculation
    confidence_margin_scale: float = 2.0
    confidence_agreement_weight: float = 0.3
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.rv_window < 5:
            raise ValueError("rv_window must be at least 5")
        if self.vol_lookback_default < self.vol_lookback_min:
            raise ValueError("vol_lookback_default must be >= vol_lookback_min")
        if self.vol_lookback_default > self.vol_lookback_max:
            raise ValueError("vol_lookback_default must be <= vol_lookback_max")
        if self.drawdown_window < 10:
            raise ValueError("drawdown_window must be at least 10")
        if self.trend_window < 10:
            raise ValueError("trend_window must be at least 10")
        if not 0 <= self.confidence_gate <= 1:
            raise ValueError("confidence_gate must be in [0, 1]")
        if self.min_regime_duration < 1:
            raise ValueError("min_regime_duration must be at least 1")
        if self.z_vol_stress_exit > self.z_vol_stress_enter:
            raise ValueError("z_vol_stress_exit must be <= z_vol_stress_enter for hysteresis")
        if self.drawdown_stress_exit < self.drawdown_stress_enter:
            raise ValueError("drawdown_stress_exit must be >= drawdown_stress_enter for hysteresis")
    
    @classmethod
    def conservative(cls) -> "RegimeConfig":
        """
        Factory for conservative config with wider hysteresis and longer min duration.
        Use this to reduce regime flip-flopping in live trading.
        """
        return cls(
            z_vol_stress_enter=2.0,
            z_vol_stress_exit=1.2,
            drawdown_stress_enter=-0.12,
            drawdown_stress_exit=-0.08,
            confidence_gate=0.7,
            min_regime_duration=10,
        )
    
    @classmethod
    def sensitive(cls) -> "RegimeConfig":
        """
        Factory for sensitive config with tighter thresholds.
        Use this for faster regime detection in research/backtesting.
        """
        return cls(
            z_vol_stress_enter=1.2,
            z_vol_stress_exit=0.8,
            drawdown_stress_enter=-0.08,
            drawdown_stress_exit=-0.05,
            confidence_gate=0.5,
            min_regime_duration=5,
        )
