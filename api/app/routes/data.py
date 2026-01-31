from fastapi import APIRouter, Query
from app.schemas.responses import SymbolsResponse, CandlesResponse
from app.deps import get_engine_client

router = APIRouter()

@router.get("/symbols", response_model=SymbolsResponse)
def list_symbols():
    engine = get_engine_client()
    return SymbolsResponse(symbols=engine.list_symbols())

@router.get("/candles", response_model=CandlesResponse)
def get_candles(
    symbol: str = Query(..., min_length=1),
    tf: str = Query("1m", description="timeframe, e.g. 1m, 5m, 1h, 1d"),
    start: str | None = Query(None, description="ISO date/time"),
    end: str | None = Query(None, description="ISO date/time"),
    limit: int = Query(200, ge=1, le=5000),
):
    engine = get_engine_client()
    candles = engine.get_candles(symbol=symbol, tf=tf, start=start, end=end, limit=limit)
    return CandlesResponse(symbol=symbol, tf=tf, candles=candles)