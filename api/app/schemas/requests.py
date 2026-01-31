from pydantic import BaseModel, Field
from typing import Literal, Optional

class BacktestRequest(BaseModel):
    strategy: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1)
    tf: str = Field("1m", description="timeframe")
    start: str = Field(..., description="ISO date/time")
    end: str = Field(..., description="ISO date/time")
    initial_cash: float = Field(10_000.0, gt=0)

class PlaceOrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    side: Literal["buy", "sell"]
    qty: float = Field(..., gt=0)
    order_type: Literal["market", "limit"] = "market"
    limit_price: Optional[float] = Field(None, gt=0)