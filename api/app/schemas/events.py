from pydantic import BaseModel, Field
from typing import Any, Dict

class Event(BaseModel):
    type: str = Field(..., description="Event type, e.g. bar, pnl_update, order_filled")
    ts: str = Field(..., description="ISO timestamp in UTC")
    data: Dict[str, Any] = Field(default_factory=dict)