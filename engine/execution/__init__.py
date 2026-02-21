"""
Execution engine module.

Provides broker adapters, order types, risk management, and the
top-level :class:`ExecutionEngine` that ties them together.
"""

from .order_types import Order, Fill, Side, OrderType, OrderStatus
from .broker_base import BrokerBase
from .broker_paper import PaperBroker
from .broker_alpaca import AlpacaBroker, AlpacaBrokerConfig
from .execution_engine import ExecutionEngine
from .risk import RiskManager, RiskLimits
from .costs import CostModel, CommissionModelConfig, SlippageModelConfig

__all__ = [
    "Order",
    "Fill",
    "Side",
    "OrderType",
    "OrderStatus",
    "BrokerBase",
    "PaperBroker",
    "AlpacaBroker",
    "AlpacaBrokerConfig",
    "ExecutionEngine",
    "RiskManager",
    "RiskLimits",
    "CostModel",
    "CommissionModelConfig",
    "SlippageModelConfig",
]
