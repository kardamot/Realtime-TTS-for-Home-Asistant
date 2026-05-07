from __future__ import annotations

import fnmatch
import os
import re
from typing import Any

import aiohttp

from app.core.config_store import ConfigStore
from app.core.log_bus import LogBus


def _scope_items(value: str) -> list[str]:
    return [item.strip().lower() for item in re.split(r"[\s,]+", value or "") if item.strip()]


def _allowed_entity_patterns(cfg: dict[str, Any]) -> list[str]:
    value = cfg.get("allowed_entities") or cfg.get("exposed_entities") or ""
    return _scope_items(str(value))


_TR_TRANSLATION_TABLE = str.maketrans(
    {
        "\u00e7": "c",
        "\u011f": "g",
        "\u0131": "i",
        "\u00f6": "o",
        "\u015f": "s",
        "\u00fc": "u",
        "\u00c7": "C",
        "\u011e": "G",
        "\u0130": "I",
        "\u00d6": "O",
        "\u015e": "S",
        "\u00dc": "U",
    }
)


def _normalize_tr(text: str) -> str:
    return text.translate(_TR_TRANSLATION_TABLE).lower()


class HomeAssistantBridge:
    def __init__(self, config_store: ConfigStore, log_bus: LogBus) -> None:
        self._config_store = config_store
        self._log_bus = log_bus

    async def status(self) -> dict[str, Any]:
        cfg = await self._cfg()
        token = self._token()
        if not cfg.get("enabled", True):
            return {
                "enabled": False,
                "connected": False,
                "reason": "disabled",
                "strict_allowlist": True,
                "allowlist_count": len(_allowed_entity_patterns(cfg)),
                "entity_scope": self.has_entity_scope(cfg),
            }
        if not token:
            return {
                "enabled": True,
                "connected": False,
                "reason": "missing_supervisor_token",
                "strict_allowlist": True,
                "allowlist_count": len(_allowed_entity_patterns(cfg)),
                "entity_scope": self.has_entity_scope(cfg),
            }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(self._base_url(cfg), headers=self._headers(token)) as resp:
                    body = await resp.text()
                    return {
                        "enabled": True,
                        "connected": resp.status < 400,
                        "status": resp.status,
                        "body": body[:96],
                        "route_home_control": bool(cfg.get("route_home_control", True)),
                        "strict_allowlist": True,
                        "allowlist_count": len(_allowed_entity_patterns(cfg)),
                        "entity_scope": self.has_entity_scope(cfg),
                    }
        except Exception as exc:
            return {
                "enabled": True,
                "connected": False,
                "reason": str(exc),
                "strict_allowlist": True,
                "allowlist_count": len(_allowed_entity_patterns(cfg)),
                "entity_scope": self.has_entity_scope(cfg),
            }

    async def is_ready(self) -> bool:
        cfg = await self._cfg()
        return bool(cfg.get("enabled", True)) and bool(self._token())

    async def get_state(self, entity_id: str) -> dict[str, Any] | None:
        cfg = await self._cfg()
        self.assert_entity_allowed(entity_id, cfg)
        async with self._session() as session:
            async with session.get(f"{self._base_url(cfg)}/states/{entity_id}", headers=self._headers()) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                return await resp.json()

    async def list_states(self, domain: str = "", limit: int = 64) -> list[dict[str, Any]]:
        cfg = await self._cfg()
        async with self._session() as session:
            async with session.get(f"{self._base_url(cfg)}/states", headers=self._headers()) as resp:
                resp.raise_for_status()
                raw_states = await resp.json()

        domain = domain.strip().lower()
        slimmed: list[dict[str, Any]] = []
        for item in raw_states if isinstance(raw_states, list) else []:
            entity_id = str(item.get("entity_id") or "")
            if not entity_id:
                continue
            if domain and not entity_id.startswith(f"{domain}."):
                continue
            if not self.is_entity_allowed(entity_id, cfg):
                continue
            attributes = item.get("attributes") or {}
            slimmed.append(
                {
                    "entity_id": entity_id,
                    "state": item.get("state"),
                    "friendly_name": attributes.get("friendly_name", ""),
                }
            )
            if len(slimmed) >= max(1, min(int(limit or 64), 256)):
                break
        return slimmed

    async def search_states(self, query: str, domain: str = "", limit: int = 8) -> list[dict[str, Any]]:
        query_terms = [term for term in re.split(r"\s+", _normalize_tr(query)) if term]
        if not query_terms:
            return []
        states = await self.list_states(domain=domain, limit=256)
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in states:
            entity_id = str(item.get("entity_id") or "")
            friendly_name = str(item.get("friendly_name") or "")
            haystack = _normalize_tr(f"{entity_id} {friendly_name}")
            score = 0
            for term in query_terms:
                if term in haystack:
                    score += 3
                if haystack.startswith(term) or f".{term}" in haystack:
                    score += 2
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("entity_id") or "")))
        return [item for _score, item in scored[: max(1, min(int(limit or 8), 20))]]

    async def call_service(self, domain: str, service: str, data: dict[str, Any] | None = None) -> Any:
        cfg = await self._cfg()
        entity_ids = self._service_entity_ids(data)
        if not entity_ids:
            raise PermissionError("Home Assistant service calls require one or more allowlisted entity_id values.")
        for entity_id in entity_ids:
            self.assert_entity_allowed(entity_id, cfg)
        async with self._session() as session:
            async with session.post(
                f"{self._base_url(cfg)}/services/{domain}/{service}",
                headers=self._headers(),
                json=data or {},
            ) as resp:
                resp.raise_for_status()
                if resp.content_type == "application/json":
                    return await resp.json()
                return await resp.text()

    async def conversation(self, text: str, language: str = "", conversation_id: str = "") -> dict[str, Any]:
        cfg = await self._cfg()
        if not bool(cfg.get("unsafe_allow_conversation_tool", False)):
            raise PermissionError(
                "Home Assistant conversation is disabled because it cannot be constrained to the Alice entity allowlist."
            )
        payload: dict[str, Any] = {
            "text": text,
            "language": language or str(cfg.get("conversation_language") or "tr"),
        }
        if cfg.get("conversation_agent_id"):
            payload["agent_id"] = str(cfg["conversation_agent_id"])
        if conversation_id:
            payload["conversation_id"] = conversation_id
        async with self._session() as session:
            async with session.post(f"{self._base_url(cfg)}/conversation/process", headers=self._headers(), json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()

    @staticmethod
    def extract_conversation_speech(result: dict[str, Any]) -> str:
        response = result.get("response") or {}
        speech = response.get("speech") or {}
        plain = speech.get("plain") or {}
        if isinstance(plain, dict) and isinstance(plain.get("speech"), str):
            return plain["speech"].strip()
        ssml = speech.get("ssml") or {}
        if isinstance(ssml, dict) and isinstance(ssml.get("speech"), str):
            return ssml["speech"].strip()
        return ""

    async def should_route_home_control(self, text: str) -> bool:
        cfg = await self._cfg()
        if not bool(cfg.get("route_home_control", True)) or not await self.is_ready():
            return False
        if not self.has_entity_scope(cfg) or not bool(cfg.get("unsafe_allow_conversation_tool", False)):
            return False
        normalized = _normalize_tr(text)
        weather_terms = ["hava", "derece", "sicaklik", "nem", "ruzgar", "yagmur"]
        device_terms = ["isik", "lamba", "priz", "klima", "perde", "panjur", "isitici", "fan", "sensor", "kamera"]
        action_terms = ["ac", "kapat", "yak", "sondur", "calistir", "durdur", "ayarla", "durum", "kac"]
        return any(term in normalized for term in weather_terms) or (
            any(term in normalized for term in device_terms) and any(term in normalized for term in action_terms)
        )

    def has_entity_scope(self, cfg: dict[str, Any]) -> bool:
        return bool(_allowed_entity_patterns(cfg))

    def is_entity_allowed(self, entity_id: str, cfg: dict[str, Any]) -> bool:
        entity_id = (entity_id or "").strip().lower()
        if not entity_id:
            return False
        return any(fnmatch.fnmatch(entity_id, pattern) for pattern in _allowed_entity_patterns(cfg))

    def assert_entity_allowed(self, entity_id: str, cfg: dict[str, Any]) -> None:
        if not self.is_entity_allowed(entity_id, cfg):
            raise PermissionError(f"Entity is not allowlisted for Alice Control Panel: {entity_id}")

    def _service_entity_ids(self, data: dict[str, Any] | None) -> list[str]:
        if not isinstance(data, dict):
            return []
        values: list[str] = []
        target = data.get("target") if isinstance(data.get("target"), dict) else {}
        for raw in (data.get("entity_id"), target.get("entity_id")):
            if isinstance(raw, str):
                values.extend(item.strip() for item in raw.split(",") if item.strip())
            elif isinstance(raw, list):
                values.extend(str(item).strip() for item in raw if str(item).strip())
        return values

    async def _cfg(self) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=True)
        value = config.get("ha_bridge", {}) if isinstance(config, dict) else {}
        return value if isinstance(value, dict) else {}

    def _token(self) -> str:
        return os.environ.get("SUPERVISOR_TOKEN", "").strip()

    def _headers(self, token: str | None = None) -> dict[str, str]:
        active_token = token if token is not None else self._token()
        headers = {"Accept": "application/json"}
        if active_token:
            headers["Authorization"] = f"Bearer {active_token}"
        return headers

    def _base_url(self, cfg: dict[str, Any]) -> str:
        return str(cfg.get("api_base_url") or "http://supervisor/core/api").rstrip("/")

    def _session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20, connect=8))
