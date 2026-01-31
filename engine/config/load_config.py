from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Any, Dict

import yaml


# ---------- dataclasses for execution config ----------

OrderTypeStr = Literal["MARKET", "LIMIT"]
FillOnStr = Literal["immediate", "next_poll"]
LimitFillRuleStr = Literal["touch", "bar_high_low"]
CommissionModelStr = Literal["none", "fixed_per_order", "per_share"]
SlippageModelStr = Literal["none", "bps"]


@dataclass(frozen=True)
class ExecutionConfig:
    default_order_type: OrderTypeStr = "MARKET"
    time_in_force: str = "DAY"


@dataclass(frozen=True)
class PaperFillConfig:
    fill_on: FillOnStr = "next_poll"
    use_spread_for_market: bool = True
    extra_slippage_bps: float = 2.0
    allow_partial_fills: bool = False
    partial_fill_ratio: float = 0.6
    limit_fill_rule: LimitFillRuleStr = "touch"
    max_fill_delay_polls: int = 0


@dataclass(frozen=True)
class CommissionConfig:
    model: CommissionModelStr = "none"
    fixed_per_order: float = 0.0
    per_share: float = 0.0
    min_commission: float = 0.0


@dataclass(frozen=True)
class SlippageConfig:
    model: SlippageModelStr = "none"
    bps: float = 0.0


@dataclass(frozen=True)
class CostsConfig:
    commission: CommissionConfig = CommissionConfig()
    slippage: SlippageConfig = SlippageConfig()


@dataclass(frozen=True)
class ExecutionYamlConfig:
    execution: ExecutionConfig = ExecutionConfig()
    paper: PaperFillConfig = PaperFillConfig()
    costs: CostsConfig = CostsConfig()


# ---------- dataclasses for risk config ----------

@dataclass(frozen=True)
class RiskConfig:
    max_notional_per_trade: float = 10_000.0
    max_position_notional: float = 30_000.0
    min_cash_buffer: float = 1_000.0
    max_open_orders: int = 20
    allow_short: bool = False


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {p} must contain a mapping/object.")
    return data


def load_execution_config(path: str | Path) -> ExecutionYamlConfig:
    raw = _load_yaml(path)

    ex_raw = (raw.get("execution") or {})
    paper_raw = (raw.get("paper") or {})
    costs_raw = (raw.get("costs") or {})
    comm_raw = (costs_raw.get("commission") or {})
    slip_raw = (costs_raw.get("slippage") or {})

    return ExecutionYamlConfig(
        execution=ExecutionConfig(**ex_raw),
        paper=PaperFillConfig(**paper_raw),
        costs=CostsConfig(
            commission=CommissionConfig(**comm_raw),
            slippage=SlippageConfig(**slip_raw),
        ),
    )


def load_risk_config(path: str | Path) -> RiskConfig:
    raw = _load_yaml(path)
    risk_raw = (raw.get("risk") or {})
    return RiskConfig(**risk_raw)