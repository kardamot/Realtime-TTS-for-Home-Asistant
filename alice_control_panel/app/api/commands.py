from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.core.auth import require_request_auth


router = APIRouter(tags=["commands"])

SERVER_COMMANDS = {
    "restart_stt",
    "restart_tts",
    "reload_prompt",
    "clear_logs",
    "safe_mode_on",
    "safe_mode_off",
}


@router.post("/api/command")
async def command(payload: dict[str, Any], request: Request, _: None = Depends(require_request_auth)) -> dict[str, Any]:
    name = str(payload.get("command") or "").strip()
    args = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    if name in SERVER_COMMANDS:
        return await _run_server_command(name, args, request)
    return await request.app.state.esp_client.send_command(name, args)


async def _run_server_command(name: str, _: dict[str, Any], request: Request) -> dict[str, Any]:
    app = request.app
    if name == "restart_stt":
        return {"ok": True, "command": name, "status": await app.state.stt_manager.restart()}
    if name == "restart_tts":
        return {"ok": True, "command": name, "status": await app.state.voice_pipeline.restart_tts()}
    if name == "reload_prompt":
        return {"ok": True, "command": name, "status": await app.state.voice_pipeline.reload_prompt()}
    if name == "clear_logs":
        await app.state.log_bus.clear()
        await app.state.log_bus.emit("INFO", "SYSTEM", "Logs cleared by command")
        return {"ok": True, "command": name}
    if name == "safe_mode_on":
        cfg = await app.state.config_store.update({"safe_mode": True})
        await app.state.log_bus.emit("WARN", "SYSTEM", "Safe mode enabled")
        return {"ok": True, "command": name, "safe_mode": cfg["safe_mode"]}
    if name == "safe_mode_off":
        cfg = await app.state.config_store.update({"safe_mode": False})
        await app.state.log_bus.emit("INFO", "SYSTEM", "Safe mode disabled")
        return {"ok": True, "command": name, "safe_mode": cfg["safe_mode"]}
    return {"ok": False, "command": name, "message": "Unknown server command"}

