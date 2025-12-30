import pandas as pd

from features.transforms import log_return, rolling_zscore
from features.indicators import sma, ema, rsi, atr


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns expected:
      - timestamp (datetime64[ns, UTC] recommended)
      - o,h,l,c,v (Polygon aggs)
    Output:
      - features dataframe indexed by timestamp
    """
    x = df.copy()

    # Ensure time index
    if "timestamp" not in x.columns:
        raise ValueError("df must contain a 'timestamp' column")
    x = x.sort_values("timestamp")
    x = x.set_index("timestamp")

    close = x["c"]
    high = x["h"]
    low = x["l"]
    vol = x["v"]

    feats = pd.DataFrame(index=x.index)
    feats.index = pd.DatetimeIndex(pd.to_datetime(feats.index))
    # Returns
    feats["ret_1m"] = close.pct_change()
    feats["logret_1m"] = log_return(close)

    # Rolling volatility (std of returns)
    feats["vol_20m"] = feats["ret_1m"].rolling(20).std(ddof=0)
    feats["vol_60m"] = feats["ret_1m"].rolling(60).std(ddof=0)

    # Trend / mean reversion features
    feats["sma_20"] = sma(close, 20)
    feats["sma_60"] = sma(close, 60)
    feats["ema_20"] = ema(close, 20)

    feats["dist_sma_20"] = (close - feats["sma_20"]) / feats["sma_20"]
    feats["z_close_60"] = rolling_zscore(close, 60)

    # Momentum indicators
    feats["rsi_14"] = rsi(close, 14)

    # Range / volatility indicator
    feats["atr_14"] = atr(high, low, close, 14)

    # Volume features
    feats["vol_z_60"] = rolling_zscore(vol, 60)
    feats["vol_sma_60"] = vol.rolling(60).mean()

    # Time features (useful for intraday seasonality)
    
    feats["hour"] = feats.index.hour
    feats["minute"] = feats.index.minute
    feats["dayofweek"] = feats.index.dayofweek

    # Clean up (drop rows with NaNs caused by rolling windows)
    feats = feats.dropna()

    return feats