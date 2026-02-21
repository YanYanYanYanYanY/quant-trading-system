"""
Long-only rank-weighted portfolio constructor.

Turns routed (non-negative) alpha scores into target portfolio weights
subject to :class:`PortfolioConstraints`.

Algorithm overview
------------------
1. **Extract** candidate scores from the :class:`AlphaVector`, keeping
   only symbols that are in *universe* and above ``score_floor``.
1b. **Risk overlay** (optional) — multiply each candidate score by a
   per-symbol risk multiplier clamped to ``[0, 1]``.  Symbols whose
   adjusted score drops to ``<= score_floor`` are removed.
2. **Select** symbols via ``top_n`` or ``top_quantile`` mode, capped to
   ``max_positions`` and floored to ``min_positions``.
3. **Weight** — raw weights proportional to score, normalised to
   ``gross_target``.
4. **Cap** each weight at ``max_weight`` and iteratively redistribute
   excess to uncapped positions.
5. **Floor** — drop any position whose weight falls below
   ``min_weight`` and renormalise.
6. **Rebalance band** (optional) — suppress small weight changes vs
   ``current_weights`` to reduce turnover.
7. **Turnover limit** (optional) — scale trades so total turnover
   ``sum(|Δw|)`` stays within the limit.

Sorting is deterministic: ties in score are broken alphabetically by
symbol so that results are reproducible across runs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..alpha.types import AlphaVector
from .types import (
    PortfolioConstraints,
    TargetPortfolio,
    clip_weights,
    enforce_max_positions,
    normalize_positive,
)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class LongOnlyRankWeightConstructor:
    """
    Score-proportional long-only portfolio constructor.

    Implements the :class:`PortfolioConstructor` protocol.  Weights
    are proportional to the routed alpha score, subject to max/min
    weight caps, position-count limits, and optional turnover
    controls.

    Parameters
    ----------
    None — all behaviour is driven by the :class:`PortfolioConstraints`
    passed to :meth:`build_targets`.

    Examples
    --------
    >>> builder = LongOnlyRankWeightConstructor()
    >>> target = builder.build_targets(
    ...     asof=pd.Timestamp("2025-06-15"),
    ...     alpha=alpha_vec,
    ...     universe=["AAPL", "MSFT", "GOOG"],
    ...     constraints=PortfolioConstraints(max_positions=20),
    ... )
    >>> target.n_positions
    3
    """

    # ------------------------------------------------------------------
    # Public API (PortfolioConstructor protocol)
    # ------------------------------------------------------------------

    def build_targets(
        self,
        asof: pd.Timestamp,
        alpha: AlphaVector,
        universe: List[str],
        constraints: PortfolioConstraints,
        current_weights: Optional[Dict[str, float]] = None,
        risk_multiplier: Optional[Dict[str, float]] = None,
    ) -> TargetPortfolio:
        """
        Build a target weight vector from alpha scores.

        Parameters
        ----------
        asof : pd.Timestamp
            Rebalance date.
        alpha : AlphaVector
            Scored universe — every ``AlphaScore.score`` should be
            ``>= 0`` (post long-only routing).
        universe : list[str]
            Eligible symbols.  Only symbols present in both *alpha*
            and *universe* are considered.
        constraints : PortfolioConstraints
            Position-count, weight, and turnover limits.
        current_weights : dict[str, float] or None
            Existing portfolio weights.  ``None`` on the first
            rebalance.  Required for rebalance-band and turnover-
            limit features.
        risk_multiplier : dict[str, float] or None
            Optional per-symbol risk overlay.  Each value is clamped
            to ``[0, 1]`` and multiplied into the candidate's alpha
            score *before* selection and weighting.  Symbols not
            present in the dict receive a multiplier of ``1.0``
            (no adjustment).  Use cases include down-weighting
            high-vol names, illiquid stocks, or names that slipped
            through a regime gate.

        Returns
        -------
        TargetPortfolio
        """
        cw = current_weights or {}

        # ── 1. Extract & filter candidates ─────────────────────────────
        candidates = self._extract_candidates(alpha, universe, constraints)

        # ── 1b. Risk overlay ───────────────────────────────────────────
        if risk_multiplier is not None:
            candidates = self._apply_risk_overlay(
                candidates, risk_multiplier, constraints,
            )

        # ── 2. Select symbols ──────────────────────────────────────────
        selected = self._select_symbols(candidates, constraints)

        # ── 3. Score-proportional raw weights ──────────────────────────
        weights = self._compute_raw_weights(
            selected, candidates, constraints,
        )

        # ── 4. Cap at max_weight ───────────────────────────────────────
        weights = self._apply_max_weight(weights, constraints)

        # ── 5. Floor at min_weight ─────────────────────────────────────
        weights = self._apply_min_weight(weights, constraints)

        # ── 6. Rebalance band ──────────────────────────────────────────
        if constraints.rebalance_band > 0.0 and cw:
            weights = self._apply_rebalance_band(
                weights, cw, constraints,
            )

        # ── 7. Turnover limit ──────────────────────────────────────────
        turnover: Optional[float] = None
        if cw:
            turnover = _compute_turnover(weights, cw)
            if (
                constraints.turnover_limit is not None
                and turnover > constraints.turnover_limit
            ):
                weights = self._apply_turnover_limit(
                    weights, cw, constraints,
                )
                turnover = _compute_turnover(weights, cw)

        # ── Build result ───────────────────────────────────────────────
        gross = sum(weights.values())

        reasons: Dict[str, object] = {
            "selected_count": float(sum(1 for w in weights.values() if w > 0)),
            "dropped_count": float(len(candidates) - sum(1 for w in weights.values() if w > 0)),
            "gross_target": constraints.gross_target,
            "cash_weight": max(0.0, constraints.gross_target - gross),
        }
        if turnover is not None:
            reasons["turnover"] = turnover

        return TargetPortfolio(
            asof=asof,
            weights=weights,
            gross=gross,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Step 1: Extract candidates
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_candidates(
        alpha: AlphaVector,
        universe: List[str],
        constraints: PortfolioConstraints,
    ) -> Dict[str, float]:
        """
        Return ``{symbol: score}`` for eligible candidates.

        Filters to *universe*, drops scores ``<= score_floor``, and
        sorts by ``(-score, symbol)`` for deterministic ordering.
        """
        universe_set = set(universe)
        raw: Dict[str, float] = {}

        for sym in universe:
            alpha_entry = alpha.get(sym)
            score = alpha_entry.score if alpha_entry is not None else 0.0
            if score > constraints.score_floor:
                raw[sym] = score

        return raw

    # ------------------------------------------------------------------
    # Step 1b: Risk overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_risk_overlay(
        candidates: Dict[str, float],
        risk_multiplier: Dict[str, float],
        constraints: PortfolioConstraints,
    ) -> Dict[str, float]:
        """
        Multiply each candidate score by a clamped risk multiplier.

        Parameters
        ----------
        candidates : dict[str, float]
            ``{symbol: score}`` from step 1.
        risk_multiplier : dict[str, float]
            Per-symbol multiplier.  Values are clamped to ``[0, 1]``.
            Symbols absent from the dict keep their original score
            (implicit multiplier of ``1.0``).
        constraints : PortfolioConstraints
            Used to re-apply ``score_floor`` after adjustment — a
            symbol whose adjusted score drops to ``<= score_floor``
            is removed.

        Returns
        -------
        dict[str, float]
            Adjusted candidate scores.
        """
        adjusted: Dict[str, float] = {}
        for sym, score in candidates.items():
            m = risk_multiplier.get(sym, 1.0)
            m = max(0.0, min(1.0, m))  # clamp to [0, 1]
            adj = score * m
            if adj > constraints.score_floor:
                adjusted[sym] = adj
        return adjusted

    # ------------------------------------------------------------------
    # Step 2: Select symbols
    # ------------------------------------------------------------------

    @staticmethod
    def _select_symbols(
        candidates: Dict[str, float],
        constraints: PortfolioConstraints,
    ) -> List[str]:
        """
        Pick which symbols enter the portfolio.

        Returns a list sorted by ``(-score, symbol)`` — deterministic
        tie-breaking by ticker name.
        """
        # Deterministic sort: descending score, ascending symbol
        ranked = sorted(
            candidates.keys(),
            key=lambda s: (-candidates[s], s),
        )

        if constraints.selection_mode == "top_quantile":
            n_quantile = max(1, math.ceil(len(ranked) * constraints.top_quantile))
            n = min(n_quantile, constraints.max_positions)
        else:
            # "top_n"
            n = constraints.max_positions

        selected = enforce_max_positions(ranked, n)

        # Ensure at least min_positions if candidates allow
        # (already handled implicitly — we only have len(ranked)
        # candidates, and we take up to n of them).
        return selected

    # ------------------------------------------------------------------
    # Step 3: Raw weights
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_raw_weights(
        selected: List[str],
        candidates: Dict[str, float],
        constraints: PortfolioConstraints,
    ) -> Dict[str, float]:
        """
        Score-proportional weights normalised to ``gross_target``.

        If ``allow_cash`` is ``False`` and there are selected symbols,
        weights always sum to ``gross_target``.  If ``allow_cash`` is
        ``True`` and fewer than ``min_positions`` candidates exist,
        the gross may be less than ``gross_target`` (excess stays
        as cash).
        """
        if not selected:
            return {}

        raw = {s: candidates[s] for s in selected}

        # When allow_cash and too few candidates, scale down gross
        target = constraints.gross_target
        if (
            constraints.allow_cash
            and len(selected) < constraints.min_positions
        ):
            # Proportionally reduce gross when short on candidates
            if constraints.min_positions > 0:
                ratio = len(selected) / constraints.min_positions
                target = constraints.gross_target * ratio

        return normalize_positive(raw, target_sum=target)

    # ------------------------------------------------------------------
    # Step 4: Max-weight cap
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_max_weight(
        weights: Dict[str, float],
        constraints: PortfolioConstraints,
    ) -> Dict[str, float]:
        """
        Cap individual weights and redistribute excess.

        After clipping, the total sum is preserved (excess is spread
        proportionally among uncapped positions).  If the cap is so
        tight that ``max_positions * max_weight < gross_target``, the
        realised gross will be below ``gross_target`` when
        ``allow_cash`` is ``True``; otherwise we renormalise.
        """
        if not weights:
            return weights

        clipped = clip_weights(weights, constraints.max_weight)

        # If allow_cash is False and sum dropped, renormalise back up
        total = sum(clipped.values())
        if (
            not constraints.allow_cash
            and total > 0.0
            and total < constraints.gross_target - 1e-12
        ):
            scale = constraints.gross_target / total
            clipped = {s: w * scale for s, w in clipped.items()}
            # Re-clip in case renormalisation pushed weights over cap
            clipped = clip_weights(clipped, constraints.max_weight)

        return clipped

    # ------------------------------------------------------------------
    # Step 5: Min-weight floor
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_min_weight(
        weights: Dict[str, float],
        constraints: PortfolioConstraints,
    ) -> Dict[str, float]:
        """
        Drop positions below ``min_weight`` and renormalise.

        Symbols whose weight falls below the floor are removed.  The
        freed weight is redistributed proportionally among survivors
        (respecting ``max_weight`` via a final clip pass).
        """
        if constraints.min_weight <= 0.0 or not weights:
            return weights

        survivors = {
            s: w for s, w in weights.items()
            if w >= constraints.min_weight
        }

        if not survivors:
            return {}

        dropped_sum = sum(
            w for w in weights.values()
            if w < constraints.min_weight
        )

        if dropped_sum <= 0.0:
            return weights

        # Redistribute dropped weight among survivors
        survivor_total = sum(survivors.values())
        if survivor_total > 0.0:
            scale = (survivor_total + dropped_sum) / survivor_total
            survivors = {s: w * scale for s, w in survivors.items()}

        # Re-clip to max_weight after redistribution
        survivors = clip_weights(survivors, constraints.max_weight)

        return survivors

    # ------------------------------------------------------------------
    # Step 6: Rebalance band
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rebalance_band(
        weights: Dict[str, float],
        current_weights: Dict[str, float],
        constraints: PortfolioConstraints,
    ) -> Dict[str, float]:
        """
        Suppress small weight changes to reduce turnover.

        For each position, if ``|w_new - w_old| < rebalance_band``
        the target weight is set to the *current* weight.  New
        entries (not in current_weights) and full exits (new weight
        == 0) are never suppressed.

        After banding, the weights are renormalised to preserve the
        target gross (with a final max-weight clip).
        """
        band = constraints.rebalance_band
        banded: Dict[str, float] = {}

        for sym, w_new in weights.items():
            w_old = current_weights.get(sym, 0.0)

            # New entry or full exit → always allow
            if w_old == 0.0 or w_new == 0.0:
                banded[sym] = w_new
                continue

            if abs(w_new - w_old) < band:
                banded[sym] = w_old  # keep current weight
            else:
                banded[sym] = w_new

        # Renormalise to desired gross
        total = sum(banded.values())
        if total > 0.0:
            target = min(constraints.gross_target, total)
            banded = normalize_positive(banded, target_sum=target)
            banded = clip_weights(banded, constraints.max_weight)

        return banded

    # ------------------------------------------------------------------
    # Step 7: Turnover limit
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_turnover_limit(
        weights: Dict[str, float],
        current_weights: Dict[str, float],
        constraints: PortfolioConstraints,
    ) -> Dict[str, float]:
        """
        Scale trades so total turnover stays within the limit.

        Computes ``w_limited = w_old + k * (w_new - w_old)`` where
        ``k = turnover_limit / turnover``.  After scaling, weights
        are floored to 0 (long-only) and a final max-weight clip
        is applied.
        """
        limit = constraints.turnover_limit
        if limit is None:
            return weights

        turnover = _compute_turnover(weights, current_weights)
        if turnover <= 0.0 or turnover <= limit:
            return weights

        k = limit / turnover

        all_syms = set(weights) | set(current_weights)
        scaled: Dict[str, float] = {}

        for sym in all_syms:
            w_new = weights.get(sym, 0.0)
            w_old = current_weights.get(sym, 0.0)
            w_limited = w_old + k * (w_new - w_old)
            w_limited = max(w_limited, 0.0)  # long-only floor
            if w_limited > 0.0:
                scaled[sym] = w_limited

        # Final clip
        if scaled:
            scaled = clip_weights(scaled, constraints.max_weight)

        return scaled


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compute_turnover(
    new_weights: Dict[str, float],
    old_weights: Dict[str, float],
) -> float:
    """
    Compute total turnover ``sum(|w_new - w_old|)`` across all symbols.

    Symbols present in only one dict are treated as 0 in the other.

    Parameters
    ----------
    new_weights : dict[str, float]
        Proposed weights.
    old_weights : dict[str, float]
        Current weights.

    Returns
    -------
    float
        Total absolute weight change.
    """
    all_syms = set(new_weights) | set(old_weights)
    return sum(
        abs(new_weights.get(s, 0.0) - old_weights.get(s, 0.0))
        for s in all_syms
    )
