from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.core.auth import require_request_auth


router = APIRouter(prefix="/api/esp", tags=["esp"])


@router.get("/status")
async def esp_status(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.esp_client.status()


@router.post("/poll")
async def esp_poll(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.esp_client.poll_once()


@router.get("/config")
async def esp_get_config(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.esp_client.get_config()


@router.post("/config")
async def esp_update_config(payload: dict[str, Any], request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.esp_client.update_config(payload)


@router.post("/command")
async def esp_command(payload: dict[str, Any], request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    command = str(payload.get("command") or "")
    return await request.app.state.esp_client.send_command(command, payload.get("payload") or {})

