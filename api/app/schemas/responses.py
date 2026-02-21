from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class SymbolsResponse(BaseModel):
    symbols: List[str]


class Candle(BaseModel):
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlesResponse(BaseModel):
    symbol: str
    tf: str
    candles: List[Candle]


class StrategiesResponse(BaseModel):
    strategies: List[str]


class BacktestStartResponse(BaseModel):
    job_id: str


class BacktestStatusResponse(BaseModel):
    job_id: str
    state: str
    progress: float = Field(0.0, ge=0.0, le=1.0)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class StatusResponse(BaseModel):
    mode: str
    engine_ok: bool
    broker_connected: bool = False
    ws_clients: int = 0
    last_event_ts: Optional[str] = None


class AccountResponse(BaseModel):
    equity: Optional[float] = None
    cash: Optional[float] = None
    buying_power: Optional[float] = None
    portfolio_value: Optional[float] = None
    daily_pnl: Optional[float] = None


class PositionsResponse(BaseModel):
    positions: List[Dict[str, Any]]


class OrdersResponse(BaseModel):
    orders: List[Dict[str, Any]]


class PlaceOrderResponse(BaseModel):
    order: Dict[str, Any]


class ClockResponse(BaseModel):
    is_open: bool = False
    next_open: Optional[str] = None
    next_close: Optional[str] = None
    timestamp: Optional[str] = None


class FlattenResponse(BaseModel):
    cancelled: int = 0
    closed: int = 0
