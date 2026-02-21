"""
Polygon.io aggregate bar downloader with retry logic.

Handles Polygon free-tier rate limits (5 req/min) via exponential
backoff on HTTP 429 responses.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

_MAX_RETRIES: int = 5
_BACKOFF_BASE: float = 15.0  # seconds; Polygon free tier resets ~60 s


def download_aggregates(
    symbol: str,
    multiplier: int,
    timespan: str,
    start: str,
    end: str,
    api_key: str,
    max_retries: int = _MAX_RETRIES,
) -> pd.DataFrame:
    """Download aggregate bars from Polygon.io.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. ``"AAPL"``).
    multiplier : int
        Bar size multiplier (e.g. ``1``).
    timespan : str
        ``'minute'``, ``'hour'``, or ``'day'``.
    start, end : str
        Date range in ``YYYY-MM-DD`` format.
    api_key : str
        Polygon API key.
    max_retries : int
        Number of retries on 429 / 5xx errors (default 5).

    Returns
    -------
    pd.DataFrame
        Columns include Polygon raw fields (``o``, ``h``, ``l``, ``c``,
        ``v``, ``t``, …) plus a derived ``timestamp`` column.
        Returns an **empty** DataFrame when the API reports no results
        (e.g. date range with no trading activity).

    Raises
    ------
    requests.HTTPError
        On non-retryable HTTP errors (4xx other than 429).
    RuntimeError
        When the API response is malformed after all retries.
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

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 429:
                wait = _BACKOFF_BASE * attempt
                log.warning(
                    "%s attempt %d/%d — 429 rate limited, "
                    "waiting %.0fs before retry",
                    symbol, attempt, max_retries, wait,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if data.get("resultsCount", 0) == 0 or "results" not in data:
                log.info("%s — no results for %s to %s", symbol, start, end)
                return pd.DataFrame()

            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            return df

        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                wait = _BACKOFF_BASE * attempt
                log.warning(
                    "%s attempt %d/%d — server error %s, retrying in %.0fs",
                    symbol, attempt, max_retries,
                    exc.response.status_code, wait,
                )
                last_exc = exc
                time.sleep(wait)
                continue
            raise

        except requests.exceptions.RequestException as exc:
            wait = _BACKOFF_BASE * attempt
            log.warning(
                "%s attempt %d/%d — network error: %s, retrying in %.0fs",
                symbol, attempt, max_retries, exc, wait,
            )
            last_exc = exc
            time.sleep(wait)
            continue

    raise RuntimeError(
        f"{symbol}: all {max_retries} download attempts failed"
    ) from last_exc
