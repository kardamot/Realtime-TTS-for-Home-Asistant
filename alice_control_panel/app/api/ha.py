from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.core.auth import require_request_auth


router = APIRouter(prefix="/api/ha", tags=["home-assistant"])


@router.get("/health")
async def ha_health(request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    return {"ok": True, "ha_bridge": await request.app.state.ha_bridge.status()}


@router.get("/states")
async def ha_list_states(
    request: Request,
    domain: str = "",
    limit: int = 64,
    _: None = Depends(require_request_auth),
) -> dict[str, Any]:
    entities = await request.app.state.ha_bridge.list_states(domain=domain, limit=limit)
    return {"ok": True, "count": len(entities), "entities": entities}


@router.get("/states/{entity_id:path}")
async def ha_get_state(entity_id: str, request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    try:
        state = await request.app.state.ha_bridge.get_state(entity_id)
    except PermissionError as exc:
        return {"ok": False, "message": str(exc)}
    return {"ok": state is not None, "entity": state}


@router.get("/search")
async def ha_search(
    request: Request,
    q: str,
    domain: str = "",
    limit: int = 8,
    _: None = Depends(require_request_auth),
) -> dict[str, Any]:
    entities = await request.app.state.ha_bridge.search_states(q, domain=domain, limit=limit)
    return {"ok": True, "count": len(entities), "entities": entities}


@router.post("/service")
async def ha_call_service(payload: dict[str, Any], request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    domain = str(payload.get("domain") or "").strip().lower()
    service = str(payload.get("service") or "").strip()
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    if not domain or not service:
        return {"ok": False, "message": "domain and service are required"}
    try:
        result = await request.app.state.ha_bridge.call_service(domain, service, data)
    except PermissionError as exc:
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "result": result}


@router.post("/conversation")
async def ha_conversation(payload: dict[str, Any], request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    text = str(payload.get("text") or "").strip()
    if not text:
        return {"ok": False, "message": "text is required"}
    try:
        result = await request.app.state.ha_bridge.conversation(
            text,
            language=str(payload.get("language") or ""),
            conversation_id=str(payload.get("conversation_id") or ""),
        )
    except PermissionError as exc:
        return {"ok": False, "message": str(exc)}
    speech = request.app.state.ha_bridge.extract_conversation_speech(result)
    return {"ok": True, "speech": speech, "result": result}
