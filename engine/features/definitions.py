"""
Feature specifications and the default catalogue.

Each :class:`FeatureSpec` is a **declarative** description of a single
feature: its name, the lookback window needed to compute it, which raw
inputs it requires, and a human-readable formula / description.  The
feature *engine* consults this catalogue to decide which rolling windows
to maintain and how to fill the cross-sectional matrix each bar.

The function :func:`default_feature_specs` returns the full list used
by the regime-aware alpha strategy.  It is intentionally a function
(not a module-level constant) so callers can filter / extend without
mutating shared state.

Naming conventions
------------------
* ``ret1``  — one-day simple return
* ``momNN`` — NN-day total return (close-to-close)
* ``smaNN`` — NN-day simple moving average of close
* ``rvNN``  — NN-day realised volatility of daily returns (annualised)
* ``advNN`` — NN-day average daily dollar volume
* ``dev20`` — deviation of close from its own 20-day mean, in vol units
* ``*_z``   — z-score variant (e.g. ``rv20_z``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Declarative description of a single feature.

    Parameters
    ----------
    name : str
        Unique feature name (also used as the column key in
        :class:`~engine.features.types.FeatureBundle`).
    lookback : int
        Minimum number of historical bars required before the feature
        can produce a finite value.  The feature engine must warm up
        at least this many bars before marking the feature ready.
    inputs : list[str]
        Raw bar fields consumed by this feature (e.g. ``["close"]``,
        ``["close", "volume"]``).  Used by the engine to validate
        that inbound bars carry the necessary data.
    description : str
        Human-readable formula and rationale.  Keep it concise but
        precise enough that a new team member could re-implement the
        feature from this string alone.

    Examples
    --------
    >>> spec = FeatureSpec(
    ...     name="ret1",
    ...     lookback=1,
    ...     inputs=["close"],
    ...     description="Daily simple return: close[t] / close[t-1] - 1.",
    ... )
    >>> spec.name
    'ret1'
    """

    name: str
    lookback: int
    inputs: List[str] = field(default_factory=lambda: ["close"])
    description: str = ""


# ---------------------------------------------------------------------------
# Default catalogue
# ---------------------------------------------------------------------------

def default_feature_specs() -> List[FeatureSpec]:
    """Return the canonical feature list for the regime-aware alpha strategy.

    The specs are grouped by function:

    **Returns / momentum**
        ``ret1``, ``mom5``, ``mom21``, ``mom63``

    **Trend filters**
        ``sma20``, ``sma50``, ``sma200``, ``trend_gap``

    **Volatility**
        ``rv20``, ``rv60``, ``rv20_z``

    **Liquidity**
        ``adv20``

    **Mean-reversion**
        ``dev20``

    **Regime helpers**
        ``max_dd_90``, ``normalized_slope``

    Each spec documents its formula in the ``description`` field.  All
    volatility figures are **annualised** (× √252).

    Returns
    -------
    list[FeatureSpec]
    """
    return [
        # ── Returns / momentum ────────────────────────────────────────
        FeatureSpec(
            name="ret1",
            lookback=1,
            inputs=["close"],
            description=(
                "Daily simple return.\n"
                "  ret1[t] = close[t] / close[t-1] - 1\n"
                "Lookback is 1 (needs yesterday's close)."
            ),
        ),
        FeatureSpec(
            name="mom5",
            lookback=5,
            inputs=["close"],
            description=(
                "5-day momentum (weekly return).\n"
                "  mom5[t] = close[t] / close[t-5] - 1"
            ),
        ),
        FeatureSpec(
            name="mom21",
            lookback=21,
            inputs=["close"],
            description=(
                "21-day momentum (~1 calendar month).\n"
                "  mom21[t] = close[t] / close[t-21] - 1"
            ),
        ),
        FeatureSpec(
            name="mom63",
            lookback=63,
            inputs=["close"],
            description=(
                "63-day momentum (~1 quarter).\n"
                "  mom63[t] = close[t] / close[t-63] - 1\n"
                "Primary signal for the momentum alpha model."
            ),
        ),

        # ── Trend filters ─────────────────────────────────────────────
        FeatureSpec(
            name="sma20",
            lookback=20,
            inputs=["close"],
            description=(
                "20-day simple moving average of close.\n"
                "  sma20[t] = mean(close[t-19 .. t])"
            ),
        ),
        FeatureSpec(
            name="sma50",
            lookback=50,
            inputs=["close"],
            description=(
                "50-day simple moving average of close.\n"
                "  sma50[t] = mean(close[t-49 .. t])\n"
                "Used as the 'short MA' in trend / regime detection."
            ),
        ),
        FeatureSpec(
            name="sma200",
            lookback=200,
            inputs=["close"],
            description=(
                "200-day simple moving average of close.\n"
                "  sma200[t] = mean(close[t-199 .. t])\n"
                "Used as the 'long MA' in trend / regime detection."
            ),
        ),
        FeatureSpec(
            name="trend_gap",
            lookback=200,
            inputs=["close"],
            description=(
                "Normalised gap between the 50-day and 200-day SMAs.\n"
                "  trend_gap[t] = (sma50[t] - sma200[t]) / sma200[t]\n"
                "Positive when short MA is above long MA (bullish cross).\n"
                "Guard: returns NaN when sma200 ≈ 0."
            ),
        ),

        # ── Volatility ────────────────────────────────────────────────
        FeatureSpec(
            name="rv20",
            lookback=20,
            inputs=["close"],
            description=(
                "20-day realised volatility (annualised).\n"
                "  rv20[t] = std(ret1[t-19 .. t], ddof=1) × √252\n"
                "Core input for regime detection and confidence scaling."
            ),
        ),
        FeatureSpec(
            name="rv60",
            lookback=60,
            inputs=["close"],
            description=(
                "60-day realised volatility (annualised).\n"
                "  rv60[t] = std(ret1[t-59 .. t], ddof=1) × √252\n"
                "Preferred vol measure for the low-vol alpha model."
            ),
        ),
        FeatureSpec(
            name="rv20_z",
            lookback=252,
            inputs=["close"],
            description=(
                "Z-score of rv20 vs its own trailing 252-day history.\n"
                "  mu  = mean(rv20[t-251 .. t])\n"
                "  sig = std(rv20[t-251 .. t], ddof=1)\n"
                "  rv20_z[t] = (rv20[t] - mu) / sig\n"
                "Guard: returns NaN when sig ≈ 0 or insufficient history.\n"
                "Used by market/stock regime detectors and mean-reversion\n"
                "alpha confidence discount."
            ),
        ),

        # ── Liquidity ─────────────────────────────────────────────────
        FeatureSpec(
            name="adv20",
            lookback=20,
            inputs=["close", "volume"],
            description=(
                "20-day average daily dollar volume.\n"
                "  dv[t] = close[t] × volume[t]\n"
                "  adv20[t] = mean(dv[t-19 .. t])\n"
                "Used for liquidity filtering and position sizing."
            ),
        ),

        # ── Mean-reversion ────────────────────────────────────────────
        FeatureSpec(
            name="dev20",
            lookback=20,
            inputs=["close"],
            description=(
                "Deviation of close from its 20-day mean, normalised by\n"
                "the product of sma20 and rv20 (vol-adjusted z-score).\n"
                "  dev20[t] = (close[t] - sma20[t]) / (sma20[t] × rv20[t])\n"
                "Guard: returns NaN when (sma20 × rv20) ≈ 0.\n"
                "Primary input for the mean-reversion alpha model.\n"
                "Note: rv20 here is the *daily* (non-annualised) std to\n"
                "keep the ratio unit-free.  If the engine only stores\n"
                "annualised rv20, divide by √252 before computing dev20."
            ),
        ),

        # ── Regime helpers ────────────────────────────────────────────
        FeatureSpec(
            name="max_dd_90",
            lookback=90,
            inputs=["close"],
            description=(
                "Rolling 90-day maximum drawdown (peak-to-trough / peak).\n"
                "  running_max = expanding_max(close[t-89 .. t])\n"
                "  dd = (close - running_max) / running_max\n"
                "  max_dd_90[t] = min(dd)\n"
                "Returns a non-positive float (e.g. -0.12 for -12%).\n"
                "Used by market and stock regime stress detectors."
            ),
        ),
        FeatureSpec(
            name="normalized_slope",
            lookback=50,
            inputs=["close"],
            description=(
                "Trend strength: OLS slope of log(close) over 50 days,\n"
                "divided by daily volatility over the same window.\n"
                "  y = log(close[t-49 .. t])\n"
                "  slope = OLS regression slope of y on [0..49]\n"
                "  daily_vol = std(diff(y), ddof=1)\n"
                "  normalized_slope[t] = slope / daily_vol\n"
                "Guard: returns NaN when daily_vol ≈ 0.\n"
                "Used by both market and stock regime trend classifiers."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def max_lookback(specs: List[FeatureSpec] | None = None) -> int:
    """Return the maximum lookback across a list of feature specs.

    If *specs* is ``None`` the default catalogue is used.

    >>> max_lookback()
    252
    """
    if specs is None:
        specs = default_feature_specs()
    if not specs:
        return 0
    return max(s.lookback for s in specs)


def required_inputs(specs: List[FeatureSpec] | None = None) -> set[str]:
    """Return the union of all raw bar inputs needed.

    >>> sorted(required_inputs())
    ['close', 'volume']
    """
    if specs is None:
        specs = default_feature_specs()
    out: set[str] = set()
    for s in specs:
        out.update(s.inputs)
    return out


def spec_by_name(
    name: str,
    specs: List[FeatureSpec] | None = None,
) -> FeatureSpec | None:
    """Look up a single spec by name, or ``None`` if not found.

    >>> spec_by_name("rv20").lookback
    20
    """
    if specs is None:
        specs = default_feature_specs()
    for s in specs:
        if s.name == name:
            return s
    return None
