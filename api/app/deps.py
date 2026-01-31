from functools import lru_cache
from app.services.engine_client import EngineClient
from app.ws.manager import WSManager

@lru_cache
def get_ws_manager() -> WSManager:
    # One shared manager per API process
    return WSManager()

@lru_cache
def get_engine_client() -> EngineClient:
    # In real life this could be:
    # - HTTP client to engine_service
    # - Redis command publisher
    # For now it's an in-memory stub.
    return EngineClient()