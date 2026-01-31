import requests
import pandas as pd


def download_aggregates(
    symbol: str,
    multiplier: int,
    timespan: str,
    start: str,
    end: str,
    api_key: str,
):
    """
    timespan: 'minute', 'hour', 'day'
    multiplier: 1, 5, 15, etc.
    """

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
        f"{multiplier}/{timespan}/{start}/{end}"
    )

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    if "results" not in data:
        raise RuntimeError(data)

    df = pd.DataFrame(data["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)

    return df