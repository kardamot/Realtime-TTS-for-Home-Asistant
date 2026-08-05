from __future__ import annotations

import json
from typing import Any

import aiohttp

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus
from app.core.prompt_store import PromptStore
from app.pipeline.llm.openai_compatible import active_llm_config


def _compact_weather_state(result: dict[str, Any]) -> dict[str, Any]:
    state = result.get("state") if isinstance(result.get("state"), dict) else {}
    attributes = state.get("attributes") if isinstance(state.get("attributes"), dict) else {}
    keys = (
        "friendly_name",
        "temperature",
        "temperature_unit",
        "apparent_temperature",
        "humidity",
        "pressure",
        "pressure_unit",
        "wind_speed",
        "wind_speed_unit",
        "wind_gust_speed",
        "wind_bearing",
        "precipitation",
        "precipitation_unit",
        "forecast",
    )
    compact_attributes = {key: attributes.get(key) for key in keys if key in attributes}
    forecast = compact_attributes.get("forecast")
    if isinstance(forecast, list):
        compact_attributes["forecast"] = forecast[:3]
    forecast_response = state.get("alice_forecast_response") if isinstance(state.get("alice_forecast_response"), dict) else {}
    compact_forecasts: dict[str, Any] = {}
    for name, response in forecast_response.items():
        if not isinstance(response, dict):
            continue
        response_data = response.get("service_response") if isinstance(response.get("service_response"), dict) else response
        compact_forecasts[name] = {}
        for entity_id, forecast_doc in response_data.items():
            if not isinstance(forecast_doc, dict):
                continue
            items = forecast_doc.get("forecast")
            compact_forecasts[name][entity_id] = {
                **{key: value for key, value in forecast_doc.items() if key != "forecast"},
                "forecast": items[:3] if isinstance(items, list) else items,
            }
    return {
        "entity_id": result.get("entity_id") or state.get("entity_id"),
        "friendly_name": attributes.get("friendly_name") or "",
        "state": state.get("state"),
        "attributes": compact_attributes,
        "forecasts": compact_forecasts,
        "fallback_speech": result.get("speech"),
    }


class HaNarrator:
    def __init__(self, config_store: ConfigStore, prompt_store: PromptStore, log_bus: LogBus) -> None:
        self._config_store = config_store
        self._prompt_store = prompt_store
        self._log_bus = log_bus

    async def narrate(self, user_text: str, result: dict[str, Any], fallback: str) -> str:
        kind = str(result.get("narration_kind") or "").strip().lower()
        # Routine controls use the local response composer so execution feedback
        # stays immediate and cannot be changed or contradicted by another model.
        if kind != "weather":
            return fallback

        config = await self._config_store.get(include_secrets=True)
        cfg = active_llm_config(config)
        provider = str(cfg.get("provider") or "openai").lower()
        if provider in {"none", "mock"}:
            return fallback
        api_key = str(cfg.get("api_key") or "").strip()
        if not api_key:
            await self._log_bus.emit("WARN", "HA", "HA narration skipped; LLM API key is empty", {"kind": kind})
            return fallback

        persona = str(cfg.get("system_prompt") or "").strip() or await self._prompt_store.active_prompt_text()
        system_prompt = (
            f"{persona}\n\n"
            "Home Assistant weather verisini Alice karakteriyle doğal Türkçeye çevir. "
            "Kısa konuş, ham state/entity/json okuma. "
            "Türkçe karakterleri doğru kullan: güneşli, rüzgar, yağmur, sıcaklık. "
            "Sıcaklık, yağış, rüzgar, nem ve hissedilen sıcaklık bilgisi varsa birlikte yorumla. "
            "Ondalıklı sayıları TTS dostu yaz: 26,3 derece yerine 26 derece civarı de; 10,8 km/h yerine 11 kilometre/saat civarı de. "
            "Rüzgar yüksekse, yağış/fırtına/kar varsa veya sıcaklık rahatsız ediciyse pratik tavsiye ekle. "
            "Bilgi yoksa uydurma; sadece eldeki veriye göre konuş. Aynı kalıba takılma. En fazla 2 cümle yaz."
        )
        payload_data = _compact_weather_state(result)
        data_label = "Home Assistant weather verisi"
        user_prompt = (
            "Kullanıcı sözü:\n"
            f"{user_text}\n\n"
            f"{data_label}:\n"
            f"{json.dumps(payload_data, ensure_ascii=False, indent=2)}\n\n"
            "Alice'in söyleyeceği doğal cevap:"
        )
        payload = {
            "model": str(cfg.get("model") or "gpt-5-mini"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": min(0.8, max(0.2, float(cfg.get("temperature", 0.6)))),
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://local.alice/addons"
            headers["X-Title"] = "Alice Control Panel"
        base_url = str(cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        await self._log_bus.emit("ERROR", "HA", "HA narration failed", {"status": resp.status, "body": body[:400], "kind": kind})
                        return fallback
                    doc = json.loads(body)
            text = str(doc.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            if not text:
                return fallback
            await self._log_bus.emit("INFO", "HA", "HA result narrated by LLM", {"chars": len(text), "provider": provider, "kind": kind})
            return text
        except Exception as exc:
            await self._log_bus.emit("ERROR", "HA", "HA narration error", {"error": str(exc), "kind": kind})
            return fallback
