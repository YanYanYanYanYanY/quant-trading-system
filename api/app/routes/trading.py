from fastapi import APIRouter, Depends, HTTPException
from app.schemas.requests import PlaceOrderRequest
from app.schemas.responses import (
    AccountResponse,
    ClockResponse,
    FlattenResponse,
    OrdersResponse,
    PlaceOrderResponse,
    PositionsResponse,
    StatusResponse,
)
from app.deps import get_engine_client, get_ws_manager
from app.services.engine_client import EngineClient
from app.ws.manager import WSManager

router = APIRouter()


@router.get("/status", response_model=StatusResponse)
def status(
    engine: EngineClient = Depends(get_engine_client),
    ws: WSManager = Depends(get_ws_manager),
):
    data = {
        **engine.get_status(),
        "ws_clients": ws.client_count,
        "last_event_ts": ws.last_event_ts,
    }
    return StatusResponse(**data)


@router.post("/start", response_model=StatusResponse)
def start_trading(
    engine: EngineClient = Depends(get_engine_client),
    ws: WSManager = Depends(get_ws_manager),
):
    try:
        engine.start_trading(ws.broadcast)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    data = {
        **engine.get_status(),
        "ws_clients": ws.client_count,
        "last_event_ts": ws.last_event_ts,
    }
    return StatusResponse(**data)


@router.post("/stop", response_model=StatusResponse)
def stop_trading(
    engine: EngineClient = Depends(get_engine_client),
    ws: WSManager = Depends(get_ws_manager),
):
    engine.stop_trading()
    data = {
        **engine.get_status(),
        "ws_clients": ws.client_count,
        "last_event_ts": ws.last_event_ts,
    }
    return StatusResponse(**data)


@router.get("/account", response_model=AccountResponse)
def account(engine: EngineClient = Depends(get_engine_client)):
    return AccountResponse(**engine.get_account())


@router.get("/clock", response_model=ClockResponse)
def clock(engine: EngineClient = Depends(get_engine_client)):
    return ClockResponse(**engine.get_clock())


@router.get("/positions", response_model=PositionsResponse)
def positions(engine: EngineClient = Depends(get_engine_client)):
    return PositionsResponse(positions=engine.get_positions())


@router.get("/orders", response_model=OrdersResponse)
def orders(engine: EngineClient = Depends(get_engine_client)):
    return OrdersResponse(orders=engine.get_orders())


@router.post("/orders", response_model=PlaceOrderResponse)
def place_order(
    req: PlaceOrderRequest,
    engine: EngineClient = Depends(get_engine_client),
    ws: WSManager = Depends(get_ws_manager),
):
    try:
        order = engine.place_order(req)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PlaceOrderResponse(order=order)


@router.post("/flatten", response_model=FlattenResponse)
def flatten_all(engine: EngineClient = Depends(get_engine_client)):
    result = engine.flatten_all()
    return FlattenResponse(**result)
