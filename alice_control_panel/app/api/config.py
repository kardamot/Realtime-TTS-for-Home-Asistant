from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from app.core.auth import require_request_auth


router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("")
async def get_config(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.config_store.get(include_secrets=False)


@router.post("")
async def update_config(payload: dict[str, Any], request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    updated = await request.app.state.config_store.update(payload)
    await request.app.state.log_bus.emit("INFO", "UI/API", "Configuration updated")
    await request.app.state.ws_hub.publish("config_updated", {"config": await request.app.state.config_store.get(False)})
    return {"ok": True, "config": updated}


@router.post("/import")
async def import_config(payload: dict[str, Any], request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    imported = payload.get("config") if "config" in payload else payload
    if not isinstance(imported, dict):
        return {"ok": False, "message": "config payload must be an object"}
    updated = await request.app.state.config_store.replace(imported)
    await request.app.state.log_bus.emit("INFO", "UI/API", "Configuration imported")
    return {"ok": True, "config": updated}


@router.get("/export")
async def export_config(
    request: Request,
    include_secrets: bool = Query(default=False),
    _: None = Depends(require_request_auth),
) -> JSONResponse:
    exported = await request.app.state.config_store.export(include_secrets=include_secrets)
    return JSONResponse(
        exported,
        headers={"Content-Disposition": "attachment; filename=alice_config.json"},
    )

