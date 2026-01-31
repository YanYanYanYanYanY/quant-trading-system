from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.deps import get_ws_manager
from app.ws.manager import WSManager

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, manager: WSManager = Depends(get_ws_manager)):
    await manager.connect(ws)
    try:
        while True:
            # Keep connection alive; optionally accept client commands
            _ = await ws.receive_text()
            # You can parse client messages here if you want (subscribe filters, etc.)
    except WebSocketDisconnect:
        manager.disconnect(ws)