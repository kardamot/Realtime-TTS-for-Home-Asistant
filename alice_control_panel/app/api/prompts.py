from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.auth import require_request_auth


router = APIRouter(prefix="/api/prompts", tags=["prompts"])


@router.get("")
async def list_prompts(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return await request.app.state.prompt_store.list_profiles()


@router.get("/{slug}")
async def get_prompt(slug: str, request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    try:
        return await request.app.state.prompt_store.get_profile(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Prompt not found") from exc


@router.post("/{slug}")
async def save_prompt(
    slug: str,
    payload: dict[str, Any],
    request: Request,
    _: None = Depends(require_request_auth),
) -> dict[str, Any]:
    doc = await request.app.state.prompt_store.save_profile(slug, payload)
    await request.app.state.log_bus.emit("INFO", "UI/API", "Prompt saved", {"slug": slug})
    return {"ok": True, "prompt": doc}


@router.post("")
async def create_prompt(payload: dict[str, Any], request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    doc = await request.app.state.prompt_store.create_profile(payload)
    await request.app.state.log_bus.emit("INFO", "UI/API", "Prompt created", {"slug": doc["slug"]})
    return {"ok": True, "prompt": doc}


@router.post("/{slug}/copy")
async def copy_prompt(
    slug: str,
    payload: dict[str, Any],
    request: Request,
    _: None = Depends(require_request_auth),
) -> dict[str, Any]:
    doc = await request.app.state.prompt_store.copy_profile(slug, str(payload.get("name") or f"{slug} copy"))
    await request.app.state.log_bus.emit("INFO", "UI/API", "Prompt copied", {"source": slug, "slug": doc["slug"]})
    return {"ok": True, "prompt": doc}


@router.delete("/{slug}")
async def delete_prompt(slug: str, request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    try:
        await request.app.state.prompt_store.delete_profile(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    config = await request.app.state.config_store.get(include_secrets=True)
    if str(config.get("prompts", {}).get("active_profile") or "") == slug:
        await request.app.state.config_store.set_active_prompt("alice")
    await request.app.state.log_bus.emit("WARN", "UI/API", "Prompt deleted", {"slug": slug})
    return {"ok": True}


@router.post("/{slug}/activate")
async def activate_prompt(slug: str, request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    try:
        await request.app.state.prompt_store.activate(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Prompt not found") from exc
    await request.app.state.log_bus.emit("INFO", "PIPELINE", "Active prompt changed", {"slug": slug})
    return {"ok": True, "active_profile": slug}
