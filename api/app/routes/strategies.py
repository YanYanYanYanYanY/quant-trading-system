from fastapi import APIRouter
from app.schemas.responses import StrategiesResponse
from app.deps import get_engine_client

router = APIRouter()

@router.get("", response_model=StrategiesResponse)
def list_strategies():
    engine = get_engine_client()
    return StrategiesResponse(strategies=engine.list_strategies())