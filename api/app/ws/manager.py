import asyncio
from typing import Any, Dict, Set
from fastapi import WebSocket
import logging
from datetime import datetime, timezone

log = logging.getLogger("ws")

class WSManager:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._last_event_ts: str | None = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def last_event_ts(self) -> str | None:
        return self._last_event_ts

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        log.info("WS connected. clients=%d", self.client_count)

    def disconnect(self, ws: WebSocket) -> None:
        # no await in disconnect path
        try:
            self._clients.remove(ws)
        except KeyError:
            pass
        log.info("WS disconnected. clients=%d", self.client_count)

    def broadcast(self, event: Dict[str, Any]) -> None:
        """
        Thread-safe-ish for FastAPI background tasks: schedule async sends.
        """
        ts = event.get("ts")
        if isinstance(ts, str):
            self._last_event_ts = ts
        else:
            self._last_event_ts = datetime.now(timezone.utc).isoformat()

        # Fire and forget
        for ws in list(self._clients):
            asyncio.create_task(self._safe_send(ws, event))

    async def _safe_send(self, ws: WebSocket, event: Dict[str, Any]) -> None:
        try:
            await ws.send_json(event)
        except Exception:
            # drop broken client
            self.disconnect(ws)