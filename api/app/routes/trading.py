from fastapi import APIRouter, Depends
from app.schemas.requests import PlaceOrderRequest
from app.schemas.responses import (
    StatusResponse,
    PositionsResponse,
    OrdersResponse,
    PlaceOrderResponse,
)
from app.deps import get_engine_client, get_ws_manager
from app.services.engine_client import EngineClient
from app.ws.manager import WSManager

router = APIRouter()

@router.get("/status", response_model=StatusResponse)
def status(engine: EngineClient = Depends(get_engine_client)):
    return StatusResponse(**engine.get_status())

@router.post("/start", response_model=StatusResponse)
def start_trading(
    engine: EngineClient = Depends(get_engine_client),
    ws: WSManager = Depends(get_ws_manager),
):
    engine.start_trading(ws.broadcast)
    return StatusResponse(**engine.get_status())

@router.post("/stop", response_model=StatusResponse)
def stop_trading(engine: EngineClient = Depends(get_engine_client)):
    engine.stop_trading()
    return StatusResponse(**engine.get_status())

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
    order = engine.place_order(req)
    # Emit an event so your React WS can update instantly
    ws.broadcast({
        "type": "order_submitted",
        "ts": engine.utcnow_iso(),
        "data": order,
    })
    return PlaceOrderResponse(order=order)