from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from app.core.auth import require_request_auth


router = APIRouter(prefix="/api/mic", tags=["mic"])


@router.get("/debug")
async def mic_debug_status(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return request.app.state.voice_pipeline.mic_debug_status()


@router.get("/debug/{channel}.wav")
async def mic_debug_wav(channel: str, request: Request, _: None = Depends(require_request_auth)) -> FileResponse:
    path = request.app.state.voice_pipeline.mic_debug_capture_path(channel)
    if path is None:
        raise HTTPException(status_code=404, detail="mic debug capture not found")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=path.name,
        headers={"Cache-Control": "no-store"},
    )
