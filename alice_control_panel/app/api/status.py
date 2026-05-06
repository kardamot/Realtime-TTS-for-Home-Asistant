from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, WebSocket

from app.core.auth import auth_required, require_request_auth, require_websocket_auth
from app.system.health import system_health


router = APIRouter(tags=["status"])


@router.get("/api/auth/check")
async def auth_check(request: Request) -> dict[str, Any]:
    cfg = await request.app.state.config_store.get(include_secrets=True)
    return {"auth_required": auth_required(cfg)}


@router.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    cfg = await request.app.state.config_store.get(include_secrets=False)
    return {
        "ok": True,
        "service": "alice_control_panel",
        "version": "0.1.10",
        "safe_mode": bool(cfg.get("safe_mode")),
        "debug_logs": bool(cfg.get("debug_logs")),
        "system": system_health(),
    }


@router.get("/api/status")
async def status(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return {
        "health": await health(request),
        "esp": await request.app.state.esp_client.status(),
        "pipeline": await request.app.state.voice_pipeline.status(),
        "stt": await request.app.state.stt_manager.status(),
        "llm": await request.app.state.llm.status(),
        "tts": await request.app.state.tts_relay.status(),
        "config": await request.app.state.config_store.get(include_secrets=False),
    }


@router.websocket("/api/ws/events")
async def events_ws(websocket: WebSocket) -> None:
    if not await require_websocket_auth(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    hub = websocket.app.state.ws_hub
    queue = await hub.subscribe()
    try:
        await websocket.send_json(
            {
                "type": "snapshot",
                "payload": {
                    "health": {
                        "ok": True,
                        "service": "alice_control_panel",
                        "version": "0.1.10",
                        "system": system_health(),
                    },
                    "esp": await websocket.app.state.esp_client.status(),
                    "pipeline": await websocket.app.state.voice_pipeline.status(),
                },
            }
        )
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    finally:
        await hub.unsubscribe(queue)
