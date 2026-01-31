from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

CommissionModel = Literal["none", "fixed_per_order", "per_share"]
SlippageModel = Literal["none", "bps"]


@dataclass(frozen=True)
class CommissionModelConfig:
    model: CommissionModel = "none"
    fixed_per_order: float = 0.0
    per_share: float = 0.0
    min_commission: float = 0.0


@dataclass(frozen=True)
class SlippageModelConfig:
    model: SlippageModel = "none"
    bps: float = 0.0


class CostModel:
    def __init__(self, commission: CommissionModelConfig, slippage: SlippageModelConfig):
        self.commission_cfg = commission
        self.slippage_cfg = slippage

    def commission(self, qty: float, price: float) -> float:
        m = self.commission_cfg.model
        if m == "none":
            return 0.0
        if m == "fixed_per_order":
            return max(self.commission_cfg.fixed_per_order, self.commission_cfg.min_commission)
        if m == "per_share":
            fee = abs(qty) * self.commission_cfg.per_share
            return max(fee, self.commission_cfg.min_commission)
        raise ValueError(f"Unknown commission model: {m}")

    def slippage_per_share(self, qty: float, price_ref: float) -> float:
        """
        Returns slippage in $/share (absolute) based on reference price.
        """
        m = self.slippage_cfg.model
        if m == "none":
            return 0.0
        if m == "bps":
            # bps of reference price -> per-share dollars
            return abs(price_ref) * (self.slippage_cfg.bps / 10_000.0)
        raise ValueError(f"Unknown slippage model: {m}")