from __future__ import annotations

import hmac
from typing import Any

from fastapi import HTTPException, Request, WebSocket


def auth_config(config: dict[str, Any]) -> tuple[str, str]:
    panel = config.get("panel", {}) if isinstance(config, dict) else {}
    return str(panel.get("token") or ""), str(panel.get("password") or "")


def auth_required(config: dict[str, Any]) -> bool:
    token, password = auth_config(config)
    return bool(token or password)


def _matches(candidate: str, token: str, password: str) -> bool:
    if not candidate:
        return False
    return bool((token and hmac.compare_digest(candidate, token)) or (password and hmac.compare_digest(candidate, password)))


def request_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        request.headers.get("x-alice-token")
        or request.query_params.get("token")
        or request.cookies.get("alice_panel_token")
        or ""
    )


def websocket_token(websocket: WebSocket) -> str:
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        websocket.headers.get("x-alice-token")
        or websocket.query_params.get("token")
        or websocket.cookies.get("alice_panel_token")
        or ""
    )


async def require_request_auth(request: Request) -> None:
    store = request.app.state.config_store
    config = await store.get(include_secrets=True)
    token, password = auth_config(config)
    if not token and not password:
        return
    if not _matches(request_token(request), token, password):
        raise HTTPException(status_code=401, detail="Alice panel token required")


async def require_websocket_auth(websocket: WebSocket) -> bool:
    store = websocket.app.state.config_store
    config = await store.get(include_secrets=True)
    token, password = auth_config(config)
    if not token and not password:
        return True
    return _matches(websocket_token(websocket), token, password)

