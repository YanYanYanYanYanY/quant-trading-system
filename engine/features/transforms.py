import numpy as np
import pandas as pd


def log_return(close: pd.Series) -> pd.Series:
    return pd.Series(np.log(close)).diff()



def pct_return(close: pd.Series) -> pd.Series:
    return close.pct_change()


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window).mean()
    std = x.rolling(window).std(ddof=0)
    return (x - mean) / std