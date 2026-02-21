import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Set

from fastapi import WebSocket

log = logging.getLogger("ws")


class WSManager:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._last_event_ts: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store a reference to the main event loop (call once at startup)."""
        self._loop = loop

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
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        log.info("WS connected. clients=%d", self.client_count)

    def disconnect(self, ws: WebSocket) -> None:
        try:
            self._clients.remove(ws)
        except KeyError:
            pass
        log.info("WS disconnected. clients=%d", self.client_count)

    def broadcast(self, event: Dict[str, Any]) -> None:
        """Thread-safe broadcast: works from any thread or the event loop."""
        ts = event.get("ts")
        if isinstance(ts, str):
            self._last_event_ts = ts
        else:
            self._last_event_ts = datetime.now(timezone.utc).isoformat()

        if not self._clients:
            return

        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
                self._loop = loop
            except RuntimeError:
                return

        for ws in list(self._clients):
            coro = self._safe_send(ws, event)
            try:
                asyncio.get_running_loop()
                asyncio.create_task(coro)
            except RuntimeError:
                asyncio.run_coroutine_threadsafe(coro, loop)

    async def _safe_send(self, ws: WebSocket, event: Dict[str, Any]) -> None:
        try:
            await ws.send_json(event)
        except Exception:
            self.disconnect(ws)
