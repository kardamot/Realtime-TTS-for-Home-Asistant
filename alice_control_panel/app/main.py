from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import commands, config, esp, logs, pipeline, prompts, status
from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.paths import FRONTEND_DIST_DIR, STATIC_DIR
from app.core.prompt_store import PromptStore
from app.core.ws_hub import WsHub
from app.esp.esp_client import EspClient
from app.pipeline.llm.openai_compatible import OpenAICompatibleLlm
from app.pipeline.stt.manager import SttManager
from app.pipeline.tts.relay import TtsRelay
from app.pipeline.voice_pipeline import VoicePipeline


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
    app = FastAPI(title="Alice Control Panel", version="0.1.0", lifespan=lifespan)
    config_store = ConfigStore()
    log_bus = LogBus(maxlen=1000)
    ws_hub = WsHub()
    prompt_store = PromptStore(config_store)
    esp_client = EspClient(config_store, log_bus, ws_hub)
    llm = OpenAICompatibleLlm(config_store, prompt_store, log_bus)

    app.state.config_store = config_store
    app.state.log_bus = log_bus
    app.state.ws_hub = ws_hub
    app.state.prompt_store = prompt_store
    app.state.esp_client = esp_client
    app.state.stt_manager = SttManager(config_store, log_bus)
    app.state.llm = llm
    app.state.tts_relay = TtsRelay(config_store, log_bus)
    app.state.voice_pipeline = VoicePipeline(config_store, log_bus, ws_hub, llm, esp_client)

    app.include_router(status.router)
    app.include_router(config.router)
    app.include_router(prompts.router)
    app.include_router(logs.router)
    app.include_router(esp.router)
    app.include_router(commands.router)
    app.include_router(pipeline.router)

    static_root = STATIC_DIR if (STATIC_DIR / "index.html").exists() else FRONTEND_DIST_DIR
    if static_root.exists():
        assets = static_root / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str = ""):
        index_path = static_root / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
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
