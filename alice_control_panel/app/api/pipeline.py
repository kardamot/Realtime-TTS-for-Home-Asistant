from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, WebSocket

from app.core.auth import require_request_auth, require_websocket_auth


router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.get("")
async def pipeline_status(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.voice_pipeline.status()


@router.post("/text")
async def pipeline_text(payload: dict[str, Any], request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    text = str(payload.get("text") or "").strip()
    if not text:
        return {"ok": False, "message": "text is required"}
    status = await request.app.state.voice_pipeline.run_text(text)
    return {"ok": True, "pipeline": status}


@router.get("/stt")
async def stt_status(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.stt_manager.status()


@router.get("/llm")
async def llm_status(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.llm.status()


@router.get("/tts")
async def tts_status(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.tts_relay.status()


@router.websocket("/tts/ws")
async def tts_websocket(websocket: WebSocket) -> None:
    if not await require_websocket_auth(websocket):
        await websocket.close(code=1008)
        return
    await websocket.app.state.tts_relay.websocket_session(websocket)

