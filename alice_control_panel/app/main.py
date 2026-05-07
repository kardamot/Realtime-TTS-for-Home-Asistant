from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import commands, config, esp, ha, logs, pipeline, prompts, status
from app.core.auth import require_websocket_auth
from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.paths import FRONTEND_DIST_DIR, STATIC_DIR
from app.core.prompt_store import PromptStore
from app.core.ws_hub import WsHub
from app.esp.esp_client import EspClient
from app.pipeline.llm.openai_compatible import OpenAICompatibleLlm
from app.pipeline.realtime.openai_realtime import OpenAIRealtimeBridge
from app.pipeline.stt.manager import SttManager
from app.pipeline.tts.relay import TtsRelay
from app.pipeline.voice_pipeline import VoicePipeline
from app.system.ha_bridge import HomeAssistantBridge


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await app.state.config_store.load()
    await app.state.prompt_store.ensure_defaults()
    await app.state.log_bus.emit("INFO", "SYSTEM", "Alice Control Panel backend starting")
    await app.state.esp_client.start()
    try:
        yield
    finally:
        await app.state.esp_client.stop()
        await app.state.log_bus.emit("INFO", "SYSTEM", "Alice Control Panel backend stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="Alice Control Panel", version="0.1.37", lifespan=lifespan)
    config_store = ConfigStore()
    log_bus = LogBus(maxlen=1000)
    ws_hub = WsHub()
    prompt_store = PromptStore(config_store)
    esp_client = EspClient(config_store, log_bus, ws_hub)
    ha_bridge = HomeAssistantBridge(config_store, log_bus)
    llm = OpenAICompatibleLlm(config_store, prompt_store, log_bus)
    stt_manager = SttManager(config_store, log_bus)
    tts_relay = TtsRelay(config_store, log_bus)
    realtime_bridge = OpenAIRealtimeBridge(config_store, prompt_store, log_bus, ws_hub, tts_relay, esp_client)
    voice_pipeline = VoicePipeline(
        config_store,
        log_bus,
        ws_hub,
        llm,
        stt_manager,
        tts_relay,
        esp_client,
        ha_bridge,
        realtime_bridge,
    )
    esp_client.set_mic_stream_handler(voice_pipeline.run_audio_capture)

    app.state.config_store = config_store
    app.state.log_bus = log_bus
    app.state.ws_hub = ws_hub
    app.state.prompt_store = prompt_store
    app.state.esp_client = esp_client
    app.state.ha_bridge = ha_bridge
    app.state.realtime_bridge = realtime_bridge
    app.state.stt_manager = stt_manager
    app.state.llm = llm
    app.state.tts_relay = tts_relay
    app.state.voice_pipeline = voice_pipeline

    app.include_router(status.router)
    app.include_router(config.router)
    app.include_router(prompts.router)
    app.include_router(logs.router)
    app.include_router(esp.router)
    app.include_router(ha.router)
    app.include_router(commands.router)
    app.include_router(pipeline.router)

    static_root = STATIC_DIR if (STATIC_DIR / "index.html").exists() else FRONTEND_DIST_DIR
    if static_root.exists():
        assets = static_root / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.websocket("/tts/ws")
    async def direct_tts_ws(websocket: WebSocket) -> None:
        if not await require_websocket_auth(websocket):
            await websocket.close(code=1008)
            return
        await websocket.app.state.tts_relay.websocket_session(websocket)

    @app.websocket("/voice/ws")
    async def direct_voice_ws(websocket: WebSocket) -> None:
        if not await require_websocket_auth(websocket):
            await websocket.close(code=1008)
            return
        await websocket.app.state.voice_pipeline.live_mic_websocket(websocket)

    @app.websocket("/ws")
    async def legacy_ws(websocket: WebSocket) -> None:
        if not await require_websocket_auth(websocket):
            await websocket.close(code=1008)
            return
        mode = str(websocket.query_params.get("mode") or "voice").lower()
        if mode == "tts":
            await websocket.app.state.tts_relay.websocket_session(websocket)
            return
        await websocket.app.state.voice_pipeline.live_mic_websocket(websocket)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str = ""):
        requested_path = (static_root / full_path).resolve()
        static_root_resolved = static_root.resolve()
        no_cache = {"Cache-Control": "no-store"}
        if (
            full_path
            and requested_path.is_file()
            and str(requested_path).startswith(str(static_root_resolved))
        ):
            return FileResponse(requested_path, headers=no_cache)
        index_path = static_root / "index.html"
        if index_path.exists():
            return FileResponse(index_path, headers=no_cache)
        return HTMLResponse(
            "<!doctype html><title>Alice Control Panel</title>"
            "<body style='font-family:system-ui;background:#111827;color:#e5e7eb;padding:32px'>"
            "<h1>Alice Control Panel backend is running</h1>"
            "<p>Frontend build was not found. Run the Vite build or build the Home Assistant add-on image.</p>"
            "<p>API health: <a style='color:#93c5fd' href='/api/health'>/api/health</a></p>"
            "</body>"
        )

    return app


app = create_app()


async def _runtime_port() -> int:
    await app.state.config_store.load()
    config = await app.state.config_store.get(include_secrets=True)
    return int(os.getenv("PORT") or config.get("panel", {}).get("port") or 8099)


if __name__ == "__main__":
    port = asyncio.run(_runtime_port())
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
