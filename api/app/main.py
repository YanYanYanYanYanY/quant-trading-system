from fastapi import FastAPI
from app.core.logging import configure_logging
from app.routes import health, data, strategies, backtests, trading
from app.routes import ws

def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(title="Quant Trading API", version="0.1.0")

    # Routers
    app.include_router(health.router, tags=["health"])
    app.include_router(data.router, prefix="/data", tags=["data"])
    app.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
    app.include_router(backtests.router, prefix="/backtests", tags=["backtests"])
    app.include_router(trading.router, prefix="/trade", tags=["trading"])
    app.include_router(ws.router, tags=["ws"])

    return app

app = create_app()