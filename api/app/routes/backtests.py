from fastapi import APIRouter, BackgroundTasks, Depends
from app.schemas.requests import BacktestRequest
from app.schemas.responses import BacktestStartResponse, BacktestStatusResponse
from app.deps import get_engine_client, get_ws_manager
from app.services.engine_client import EngineClient
from app.ws.manager import WSManager

router = APIRouter()

@router.post("", response_model=BacktestStartResponse)
def start_backtest(
    req: BacktestRequest,
    bg: BackgroundTasks,
    engine: EngineClient = Depends(get_engine_client),
    ws: WSManager = Depends(get_ws_manager),
):
    # Start job
    job_id = engine.start_backtest(req)

    # Simulate engine emitting events; in real life engine emits and API just forwards
    bg.add_task(engine.simulate_backtest_run, job_id, ws.broadcast)

    return BacktestStartResponse(job_id=job_id)

@router.get("/{job_id}", response_model=BacktestStatusResponse)
def backtest_status(
    job_id: str,
    engine: EngineClient = Depends(get_engine_client),
):
    status = engine.get_backtest_status(job_id)
    return BacktestStatusResponse(**status)