from __future__ import annotations

import json
from typing import Any, AsyncIterator

import aiohttp

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.prompt_store import PromptStore


class OpenAICompatibleLlm:
    def __init__(self, config_store: ConfigStore, prompt_store: PromptStore, log_bus: LogBus) -> None:
        self._config_store = config_store
        self._prompt_store = prompt_store
        self._log_bus = log_bus

    async def status(self) -> dict[str, Any]:
        cfg = (await self._config_store.get(include_secrets=False)).get("llm", {})
        return {
            "provider": cfg.get("provider", "openai"),
            "model": cfg.get("model", "gpt-5-mini"),
            "base_url": cfg.get("base_url", ""),
            "api_key_configured": bool(cfg.get("api_key")),
            "stream": bool(cfg.get("stream", True)),
        }

    async def stream_chat(self, user_text: str) -> AsyncIterator[str]:
        config = await self._config_store.get(include_secrets=True)
        cfg = config.get("llm", {})
        provider = str(cfg.get("provider") or "openai").lower()
        if provider in {"none", "mock"}:
            yield f"Mock LLM: {user_text}"
            return
        api_key = str(cfg.get("api_key") or "").strip()
        if not api_key:
            await self._log_bus.emit("WARN", "LLM", "LLM API key is empty; returning mock response")
            yield "LLM API anahtari ayarlanmamis. Bu bir mock cevaptir."
            return
        base_url = str(cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        system_prompt = str(cfg.get("system_prompt") or "").strip() or await self._prompt_store.active_prompt_text()
        payload = {
            "model": str(cfg.get("model") or "gpt-5-mini"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": float(cfg.get("temperature", 0.6)),
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    await self._log_bus.emit("ERROR", "LLM", "LLM request failed", {"status": resp.status, "body": body[:500]})
                    raise RuntimeError(f"LLM HTTP {resp.status}: {body[:300]}")
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        doc = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = doc.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield str(delta)

