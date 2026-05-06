from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, WebSocket
from fastapi.responses import PlainTextResponse

from app.core.auth import require_request_auth, require_websocket_auth


router = APIRouter(tags=["logs"])


@router.get("/api/logs")
async def get_logs(
    request: Request,
    level: str | None = Query(default=None),
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=250, ge=1, le=1000),
    _: None = Depends(require_request_auth),
) -> dict[str, Any]:
    entries = await request.app.state.log_bus.list(level=level, category=category, search=search, limit=limit)
    return {"entries": entries}


@router.delete("/api/logs")
async def clear_logs(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    await request.app.state.log_bus.clear()
    await request.app.state.log_bus.emit("INFO", "SYSTEM", "Logs cleared")
    return {"ok": True}


@router.get("/api/logs/download")
async def download_logs(request: Request, _: None = Depends(require_request_auth)) -> PlainTextResponse:
    body = await request.app.state.log_bus.download_text()
    return PlainTextResponse(
        body,
        headers={"Content-Disposition": "attachment; filename=alice_logs.txt"},
    )


@router.websocket("/api/ws/logs")
async def logs_ws(websocket: WebSocket) -> None:
    if not await require_websocket_auth(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    bus = websocket.app.state.log_bus
    queue = await bus.subscribe()
    try:
        initial = await bus.list(limit=250)
        await websocket.send_json({"type": "snapshot", "entries": initial})
        while True:
            batch = []
            try:
                item = await asyncio.wait_for(queue.get(), timeout=20)
                batch.append(item.to_dict())
                deadline = asyncio.get_running_loop().time() + 0.1
                while asyncio.get_running_loop().time() < deadline and len(batch) < 100:
                    try:
                        batch.append(queue.get_nowait().to_dict())
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json({"type": "entries", "entries": batch})
    finally:
        await bus.unsubscribe(queue)

