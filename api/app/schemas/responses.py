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
    state: str  # queued/running/done/failed
    progress: float = Field(0.0, ge=0.0, le=1.0)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    mode: str  # stopped/paper/live
    engine_ok: bool
    ws_clients: int
    last_event_ts: Optional[str] = None

class PositionsResponse(BaseModel):
    positions: List[Dict[str, Any]]

class OrdersResponse(BaseModel):
    orders: List[Dict[str, Any]]

class PlaceOrderResponse(BaseModel):
    order: Dict[str, Any]