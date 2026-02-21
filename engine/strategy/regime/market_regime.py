"""
SPY Market Regime Detector.

Detects market regime using SPY price data only.
Outputs a stable regime label + confidence for use by the strategy layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, datetime
from enum import Enum
from typing import Dict, Optional, Tuple, Union

from .config import RegimeConfig
from .market_types import MarketRegimeLabel, MarketRegimeState


class _RegimeType(str, Enum):
    """Internal fine-grained regime labels used by the detector for hysteresis."""
    CALM_TRENDING_UP = "calm_trending_up"
    CALM_TRENDING_DOWN = "calm_trending_down"
    CHOPPY_NORMAL = "choppy_normal"
    STRESSED = "stressed"


# Map internal fine-grained labels to the public MarketRegimeLabel.
_REGIME_TO_LABEL: Dict[_RegimeType, MarketRegimeLabel] = {
    _RegimeType.CALM_TRENDING_UP: MarketRegimeLabel.CALM_TREND,
    _RegimeType.CALM_TRENDING_DOWN: MarketRegimeLabel.CALM_TREND,
    _RegimeType.CHOPPY_NORMAL: MarketRegimeLabel.CALM_RANGE,
    _RegimeType.STRESSED: MarketRegimeLabel.STRESS,
}


class MarketRegimeDetector:
    """
    Detects market regime from SPY daily OHLCV data.
    
    Features computed:
    1. Realized vol 20d (annualized)
    2. Vol z-score vs 1-3y history
    3. Max drawdown over trailing window
    4. Normalized trend slope (slope / vol)
    5. Optional: 50/200 MA crossover
    
    Regimes:
    - calm_trending_up: low vol, positive trend
    - calm_trending_down: low vol, negative trend
    - choppy_normal: normal vol, no clear trend
    - stressed: high vol and/or deep drawdown
    
    Parameters
    ----------
    config : RegimeConfig, optional
        Configuration parameters. Uses defaults if not provided.
    
    Examples
    --------
    >>> detector = MarketRegimeDetector()
    >>> state = detector.update(spy_df)
    >>> print(state.regime, state.confidence)
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        
        # Internal state for stability controls
        self._last_regime: Optional[_RegimeType] = None
        self._last_switch_date: Optional[Union[date, datetime]] = None
        self._regime_start_date: Optional[Union[date, datetime]] = None
        self._days_in_regime: int = 0
    
    def reset_state(self) -> None:
        """Reset internal state. Call this when starting a new backtest."""
        self._last_regime = None
        self._last_switch_date = None
        self._regime_start_date = None
        self._days_in_regime = 0
    
    @property
    def current_regime(self) -> Optional[MarketRegimeLabel]:
        """Get the current regime without updating."""
        if self._last_regime is None:
            return None
        return _REGIME_TO_LABEL[self._last_regime]
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all regime features from price data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least 'close' column, sorted by date ascending.
            Optionally 'high', 'low' columns. Index can be datetime or integer.
        
        Returns
        -------
        pd.DataFrame
            Original DataFrame with added feature columns:
            - log_return: daily log return
            - rv20: 20-day realized vol (annualized)
            - rv20_mean: rolling mean of rv20 for z-score
            - rv20_std: rolling std of rv20 for z-score
            - z_vol: vol z-score
            - max_dd: max drawdown over window
            - normalized_slope: trend slope / vol
            - ma_short: short MA
            - ma_long: long MA
            - ma_trend_flag: MA crossover signal
        """
        df = df.copy()
        cfg = self.config
        
        # Ensure we have required columns
        if "close" not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        close = df["close"].astype(float)
        
        # 1. Log returns
        df["log_return"] = np.log(close / close.shift(1))
        
        # 2. Realized volatility (20d, annualized)
        df["rv20"] = (
            df["log_return"]
            .rolling(window=cfg.rv_window, min_periods=cfg.rv_window)
            .std()
            * cfg.rv_annualization_factor
        )
        
        # 3. Vol z-score vs history
        # Use expanding window up to vol_lookback_default, min vol_lookback_min
        df["rv20_mean"] = (
            df["rv20"]
            .rolling(window=cfg.vol_lookback_default, min_periods=cfg.vol_lookback_min)
            .mean()
        )
        df["rv20_std"] = (
            df["rv20"]
            .rolling(window=cfg.vol_lookback_default, min_periods=cfg.vol_lookback_min)
            .std()
        )
        # Avoid division by zero
        df["z_vol"] = (df["rv20"] - df["rv20_mean"]) / df["rv20_std"].replace(0, np.nan)
        
        # 4. Max drawdown over trailing window
        df["max_dd"] = self._compute_rolling_max_drawdown(close, cfg.drawdown_window)
        
        # 5. Normalized trend slope
        df["normalized_slope"] = self._compute_normalized_slope(
            close, cfg.trend_window
        )
        
        # 6. MA crossover (optional but always computed for metrics)
        df["ma_short"] = close.rolling(window=cfg.ma_short, min_periods=cfg.ma_short).mean()
        df["ma_long"] = close.rolling(window=cfg.ma_long, min_periods=cfg.ma_long).mean()
        df["ma_trend_flag"] = np.where(
            df["ma_short"] > df["ma_long"], 1,
            np.where(df["ma_short"] < df["ma_long"], -1, 0)
        )
        
        return df
    
    def _compute_rolling_max_drawdown(
        self, close: pd.Series, window: int
    ) -> pd.Series:
        """
        Compute rolling max drawdown over trailing window.
        
        Returns negative values (e.g., -0.10 for -10% drawdown).
        """
        result = pd.Series(index=close.index, dtype=float)
        
        for i in range(len(close)):
            if i < window - 1:
                result.iloc[i] = np.nan
                continue
            
            window_data = close.iloc[max(0, i - window + 1):i + 1]
            running_max = window_data.expanding().max()
            drawdowns = (window_data - running_max) / running_max
            result.iloc[i] = drawdowns.min()
        
        return result
    
    def _compute_normalized_slope(
        self, close: pd.Series, window: int
    ) -> pd.Series:
        """
        Compute normalized slope: regression slope on log(close) / daily vol.
        
        This gives trend strength in vol-adjusted terms.
        """
        log_close = np.log(close)
        result = pd.Series(index=close.index, dtype=float)
        
        for i in range(len(close)):
            if i < window - 1:
                result.iloc[i] = np.nan
                continue
            
            # Get window data
            y = log_close.iloc[i - window + 1:i + 1].values
            x = np.arange(window)
            
            # Linear regression slope
            x_mean = x.mean()
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            
            # Daily vol over same window
            log_returns = np.diff(y)
            daily_vol = np.std(log_returns, ddof=1) if len(log_returns) > 1 else np.nan
            
            # Normalized slope
            if daily_vol > 0 and not np.isnan(daily_vol):
                result.iloc[i] = slope / daily_vol
            else:
                result.iloc[i] = np.nan
        
        return result
    
    def _classify_regime(
        self,
        z_vol: float,
        max_dd: float,
        normalized_slope: float,
        ma_trend_flag: int,
        current_regime: Optional[_RegimeType],
    ) -> Tuple[_RegimeType, float]:
        """
        Classify regime based on features with hysteresis.
        
        Returns
        -------
        Tuple[_RegimeType, float]
            (regime, confidence)
        """
        cfg = self.config
        
        # Check for stressed regime (highest priority)
        is_stressed, stress_confidence = self._check_stressed(
            z_vol, max_dd, current_regime
        )
        
        if is_stressed:
            return _RegimeType.STRESSED, stress_confidence
        
        # Check trend direction
        trend_direction, trend_confidence = self._check_trend(
            normalized_slope, ma_trend_flag, current_regime
        )
        
        # Classify based on trend
        if trend_direction > 0:
            return _RegimeType.CALM_TRENDING_UP, trend_confidence
        elif trend_direction < 0:
            return _RegimeType.CALM_TRENDING_DOWN, trend_confidence
        else:
            # No clear trend = choppy/normal
            return _RegimeType.CHOPPY_NORMAL, trend_confidence
    
    def _check_stressed(
        self,
        z_vol: float,
        max_dd: float,
        current_regime: Optional[_RegimeType],
    ) -> Tuple[bool, float]:
        """
        Check if we should be in stressed regime, with hysteresis.
        
        Returns (is_stressed, confidence).
        """
        cfg = self.config
        currently_stressed = current_regime == _RegimeType.STRESSED
        
        # Use different thresholds based on current state (hysteresis)
        if currently_stressed:
            vol_threshold = cfg.z_vol_stress_exit
            dd_threshold = cfg.drawdown_stress_exit
        else:
            vol_threshold = cfg.z_vol_stress_enter
            dd_threshold = cfg.drawdown_stress_enter
        
        # Check conditions
        vol_stressed = z_vol > vol_threshold if not np.isnan(z_vol) else False
        dd_stressed = max_dd < dd_threshold if not np.isnan(max_dd) else False
        
        is_stressed = vol_stressed or dd_stressed
        
        # Compute confidence based on margin from thresholds
        confidence = 0.5  # base confidence
        
        if is_stressed:
            # How far above stress threshold?
            vol_margin = (z_vol - vol_threshold) / cfg.confidence_margin_scale if not np.isnan(z_vol) else 0
            dd_margin = (dd_threshold - max_dd) / (abs(dd_threshold) * cfg.confidence_margin_scale) if not np.isnan(max_dd) else 0
            
            margin = max(vol_margin, dd_margin)
            confidence = min(1.0, 0.5 + margin * 0.5)
        else:
            # How far below stress threshold?
            vol_margin = (vol_threshold - z_vol) / cfg.confidence_margin_scale if not np.isnan(z_vol) else 0
            dd_margin = (max_dd - dd_threshold) / (abs(dd_threshold) * cfg.confidence_margin_scale) if not np.isnan(max_dd) else 0
            
            margin = min(vol_margin, dd_margin)
            confidence = min(1.0, 0.5 + margin * 0.5)
        
        return is_stressed, max(0.0, min(1.0, confidence))
    
    def _check_trend(
        self,
        normalized_slope: float,
        ma_trend_flag: int,
        current_regime: Optional[_RegimeType],
    ) -> Tuple[int, float]:
        """
        Determine trend direction with hysteresis.
        
        Returns (direction, confidence) where direction is +1, -1, or 0.
        """
        cfg = self.config
        
        # Current trend state for hysteresis
        currently_up = current_regime == _RegimeType.CALM_TRENDING_UP
        currently_down = current_regime == _RegimeType.CALM_TRENDING_DOWN
        
        # Slope-based trend with hysteresis
        if np.isnan(normalized_slope):
            slope_direction = 0
            slope_confidence = 0.5
        else:
            if currently_up:
                # Need to drop below exit threshold to leave uptrend
                if normalized_slope > cfg.trend_up_exit:
                    slope_direction = 1
                elif normalized_slope < cfg.trend_down_enter:
                    slope_direction = -1
                else:
                    slope_direction = 0
            elif currently_down:
                # Need to rise above exit threshold to leave downtrend
                if normalized_slope < cfg.trend_down_exit:
                    slope_direction = -1
                elif normalized_slope > cfg.trend_up_enter:
                    slope_direction = 1
                else:
                    slope_direction = 0
            else:
                # Not in a trend, use enter thresholds
                if normalized_slope > cfg.trend_up_enter:
                    slope_direction = 1
                elif normalized_slope < cfg.trend_down_enter:
                    slope_direction = -1
                else:
                    slope_direction = 0
            
            # Slope confidence based on magnitude
            slope_magnitude = abs(normalized_slope)
            slope_confidence = min(1.0, 0.5 + slope_magnitude / cfg.confidence_margin_scale * 0.5)
        
        # Combine with MA crossover if enabled
        if cfg.use_ma_crossover and ma_trend_flag != 0:
            # Agreement bonus
            if slope_direction == ma_trend_flag:
                agreement_bonus = cfg.confidence_agreement_weight
            elif slope_direction == 0:
                # MA crossover breaks tie
                slope_direction = ma_trend_flag
                agreement_bonus = 0
            else:
                # Disagreement penalty
                agreement_bonus = -cfg.confidence_agreement_weight * 0.5
            
            final_confidence = min(1.0, max(0.0, slope_confidence + agreement_bonus))
        else:
            final_confidence = slope_confidence
        
        return slope_direction, final_confidence
    
    def _apply_stability_controls(
        self,
        new_regime: _RegimeType,
        new_confidence: float,
        asof: Union[date, datetime],
    ) -> Tuple[_RegimeType, float]:
        """
        Apply confidence gate and minimum duration controls.
        
        Returns the (final_regime, final_confidence) after stability checks.
        """
        cfg = self.config
        
        # First call - no previous regime
        if self._last_regime is None:
            return new_regime, new_confidence
        
        # Same regime - just update
        if new_regime == self._last_regime:
            return new_regime, new_confidence
        
        # Attempting to switch regimes - apply controls
        
        # 1. Check minimum duration
        if self._days_in_regime < cfg.min_regime_duration:
            # Can't switch yet, stay in current regime
            return self._last_regime, new_confidence * 0.5  # Reduced confidence
        
        # 2. Check confidence gate
        if new_confidence < cfg.confidence_gate:
            # Confidence too low, stay in current regime
            return self._last_regime, new_confidence
        
        # All checks passed - allow regime switch
        return new_regime, new_confidence
    
    def update(self, df: pd.DataFrame) -> MarketRegimeState:
        """
        Update regime state using the latest data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least 'close' column, sorted by date ascending.
            Should have enough history for all feature calculations
            (at least vol_lookback_default + rv_window rows recommended).
        
        Returns
        -------
        MarketRegimeState
            Current regime state with all metrics.
        
        Raises
        ------
        ValueError
            If DataFrame is too short or missing required columns.
        """
        if len(df) < self.config.rv_window + 10:
            raise ValueError(
                f"DataFrame too short. Need at least {self.config.rv_window + 10} rows, "
                f"got {len(df)}"
            )
        
        # Compute all features
        df_features = self.compute_features(df)
        
        # Get latest row values
        latest = df_features.iloc[-1]
        
        # Extract date
        if isinstance(df_features.index, pd.DatetimeIndex):
            asof = df_features.index[-1]
            if hasattr(asof, 'date'):
                asof = asof.date() if isinstance(asof, datetime) else asof
        elif "date" in df_features.columns:
            asof = latest["date"]
        elif "timestamp" in df_features.columns:
            asof = latest["timestamp"]
        else:
            asof = date.today()
        
        # Extract features
        rv20 = latest["rv20"]
        z_vol = latest["z_vol"]
        max_dd = latest["max_dd"]
        normalized_slope = latest["normalized_slope"]
        ma_trend_flag = int(latest["ma_trend_flag"]) if not np.isnan(latest["ma_trend_flag"]) else 0
        
        # Classify regime (raw, before stability controls)
        raw_regime, raw_confidence = self._classify_regime(
            z_vol=z_vol,
            max_dd=max_dd,
            normalized_slope=normalized_slope,
            ma_trend_flag=ma_trend_flag,
            current_regime=self._last_regime,
        )
        
        # Apply stability controls
        final_regime, final_confidence = self._apply_stability_controls(
            new_regime=raw_regime,
            new_confidence=raw_confidence,
            asof=asof,
        )
        
        # Update internal state
        if final_regime != self._last_regime:
            self._last_regime = final_regime
            self._last_switch_date = asof
            self._regime_start_date = asof
            self._days_in_regime = 1
        else:
            self._days_in_regime += 1
        
        # Build scores dict
        scores: Dict = {
            "rv20": float(rv20) if not np.isnan(rv20) else None,
            "z_vol": float(z_vol) if not np.isnan(z_vol) else None,
            "max_dd": float(max_dd) if not np.isnan(max_dd) else None,
            "normalized_slope": float(normalized_slope) if not np.isnan(normalized_slope) else None,
            "ma_trend_flag": ma_trend_flag,
            "raw_regime": raw_regime.value,
            "raw_confidence": float(raw_confidence),
            "days_in_regime": self._days_in_regime,
            "regime_start_date": str(self._regime_start_date),
        }
        
        return MarketRegimeState(
            asof=pd.Timestamp(asof),
            label=_REGIME_TO_LABEL[final_regime],
            confidence=final_confidence,
            scores=scores,
        )
    
    def update_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire DataFrame and return regime for each row.
        
        Useful for backtesting. Resets state before processing.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full price history DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with regime columns added:
            - regime: regime classification
            - regime_confidence: confidence score
        """
        self.reset_state()
        
        # Compute features once
        df_features = self.compute_features(df)
        
        # Minimum rows needed before we can classify
        min_rows = max(
            self.config.rv_window,
            self.config.vol_lookback_min + self.config.rv_window,
            self.config.drawdown_window,
            self.config.trend_window,
            self.config.ma_long,
        )
        
        regimes = []
        confidences = []
        
        for i in range(len(df_features)):
            if i < min_rows - 1:
                regimes.append(None)
                confidences.append(None)
                continue
            
            # Get data up to this point
            row = df_features.iloc[i]
            
            # Extract date
            if isinstance(df_features.index, pd.DatetimeIndex):
                asof = df_features.index[i]
            else:
                asof = date.today()
            
            # Extract features
            rv20 = row["rv20"]
            z_vol = row["z_vol"]
            max_dd = row["max_dd"]
            normalized_slope = row["normalized_slope"]
            ma_trend_flag = int(row["ma_trend_flag"]) if not pd.isna(row["ma_trend_flag"]) else 0
            
            # Skip if key features are NaN
            if pd.isna(z_vol) or pd.isna(max_dd) or pd.isna(normalized_slope):
                regimes.append(None)
                confidences.append(None)
                continue
            
            # Classify
            raw_regime, raw_confidence = self._classify_regime(
                z_vol=z_vol,
                max_dd=max_dd,
                normalized_slope=normalized_slope,
                ma_trend_flag=ma_trend_flag,
                current_regime=self._last_regime,
            )
            
            # Apply stability controls
            final_regime, final_confidence = self._apply_stability_controls(
                new_regime=raw_regime,
                new_confidence=raw_confidence,
                asof=asof,
            )
            
            # Update state
            if final_regime != self._last_regime:
                self._last_regime = final_regime
                self._last_switch_date = asof
                self._regime_start_date = asof
                self._days_in_regime = 1
            else:
                self._days_in_regime += 1
            
            label = _REGIME_TO_LABEL[final_regime] if final_regime else None
            regimes.append(label.value if label else None)
            confidences.append(final_confidence)
        
        df_features["regime"] = regimes
        df_features["regime_confidence"] = confidences
        
        return df_features
