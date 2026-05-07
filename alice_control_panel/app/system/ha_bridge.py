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


def _explicit_entity_ids(cfg: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for pattern in _allowed_entity_patterns(cfg):
        if any(char in pattern for char in "*?[]"):
            continue
        if "." in pattern and pattern not in values:
            values.append(pattern)
    return values


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


_TURN_ON_TERMS = {"ac", "yak", "baslat", "calistir", "aktiflestir"}
_TURN_OFF_TERMS = {"kapat", "kapa", "sondur", "durdur", "pasiflestir"}
_TOGGLE_TERMS = {"degistir", "toggle"}
_READ_TERMS = {"durum", "durumu", "kac", "nedir", "nasil", "oku", "goster", "sicaklik", "nem", "hava"}
_DOMAIN_HINTS = {
    "isik": "light",
    "isig": "light",
    "lamba": "light",
    "lamb": "light",
    "avize": "light",
    "led": "light",
    "priz": "switch",
    "anahtar": "switch",
    "role": "switch",
    "fan": "fan",
    "vantilator": "fan",
    "perde": "cover",
    "panjur": "cover",
    "garaj": "cover",
    "klima": "climate",
    "termostat": "climate",
    "sensor": "sensor",
    "sicaklik": "sensor",
    "sicak": "sensor",
    "derece": "sensor",
    "nem": "sensor",
    "hava": "weather",
    "kilit": "lock",
}
_IGNORED_MATCH_TERMS = (
    _TURN_ON_TERMS
    | _TURN_OFF_TERMS
    | _TOGGLE_TERMS
    | _READ_TERMS
    | set(_DOMAIN_HINTS)
    | {"alice", "lutfen", "bir", "de", "da", "mi", "mu", "midir", "bana", "icin", "su", "sunu", "oradaki"}
)
_CONTROL_DOMAINS = {"light", "switch", "fan", "input_boolean", "media_player", "climate", "humidifier"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_.]+", _normalize_tr(text))


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
                "explicit_entity_count": len(_explicit_entity_ids(cfg)),
                "entity_scope": self.has_entity_scope(cfg),
            }
        if not token:
            return {
                "enabled": True,
                "connected": False,
                "reason": "missing_supervisor_token",
                "strict_allowlist": True,
                "allowlist_count": len(_allowed_entity_patterns(cfg)),
                "explicit_entity_count": len(_explicit_entity_ids(cfg)),
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
                        "explicit_entity_count": len(_explicit_entity_ids(cfg)),
                        "entity_scope": self.has_entity_scope(cfg),
                    }
        except Exception as exc:
            return {
                "enabled": True,
                "connected": False,
                "reason": str(exc),
                "strict_allowlist": True,
                "allowlist_count": len(_allowed_entity_patterns(cfg)),
                "explicit_entity_count": len(_explicit_entity_ids(cfg)),
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
        domain = domain.strip().lower()
        entity_ids = _explicit_entity_ids(cfg)
        slimmed: list[dict[str, Any]] = []
        max_items = max(1, min(int(limit or 64), 256))
        async with self._session() as session:
            for entity_id in entity_ids:
                if domain and not entity_id.startswith(f"{domain}."):
                    continue
                async with session.get(f"{self._base_url(cfg)}/states/{entity_id}", headers=self._headers()) as resp:
                    if resp.status == 404:
                        continue
                    resp.raise_for_status()
                    item = await resp.json()
                attributes = item.get("attributes") or {}
                slimmed.append(
                    {
                        "entity_id": entity_id,
                        "state": item.get("state"),
                        "friendly_name": attributes.get("friendly_name", ""),
                    }
                )
                if len(slimmed) >= max_items:
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

    async def handle_text_command(self, text: str) -> dict[str, Any]:
        cfg = await self._cfg()
        action = self._detect_action(text)
        domain_hint = self._domain_hint(text)
        if not action:
            return {"handled": False, "ok": False, "reason": "no_home_assistant_intent"}
        if not bool(cfg.get("enabled", True)):
            return {"handled": True, "ok": False, "speech": "Home Assistant baglantisi kapali."}
        if not bool(cfg.get("route_home_control", True)):
            return {"handled": False, "ok": False, "reason": "routing_disabled"}
        if not self.has_entity_scope(cfg):
            return {"handled": True, "ok": False, "speech": "Home Assistant allowlist bos. Once izin verilen entity listesini doldurmalisin."}
        if not await self.is_ready():
            return {"handled": True, "ok": False, "speech": "Home Assistant API hazir degil."}

        entity, alternatives = await self._select_entity(text, domain_hint)
        if entity is None:
            if domain_hint:
                names = ", ".join(str(item.get("friendly_name") or item.get("entity_id")) for item in alternatives[:3])
                suffix = f" Adaylar: {names}." if names else ""
                return {"handled": True, "ok": False, "speech": f"Allowlist icinde uygun Home Assistant entity bulamadim.{suffix}"}
            return {"handled": False, "ok": False, "reason": "no_matching_entity"}

        entity_id = str(entity.get("entity_id") or "")
        friendly = str(entity.get("friendly_name") or entity_id)
        domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
        if action == "read":
            state = await self.get_state(entity_id)
            speech = self._state_speech(state or entity, friendly)
            await self._log_bus.emit("INFO", "HA", "Allowlisted HA state read", {"entity_id": entity_id})
            return {"handled": True, "ok": True, "action": action, "entity_id": entity_id, "speech": speech, "state": state}

        service = self._service_for_action(domain, action)
        if not service:
            return {
                "handled": True,
                "ok": False,
                "entity_id": entity_id,
                "speech": f"{friendly} icin bu komut henuz desteklenmiyor.",
            }
        result = await self.call_service(domain, service, {"entity_id": entity_id})
        await self._log_bus.emit("INFO", "HA", "Allowlisted HA service call", {"entity_id": entity_id, "domain": domain, "service": service})
        return {
            "handled": True,
            "ok": True,
            "action": action,
            "entity_id": entity_id,
            "domain": domain,
            "service": service,
            "speech": self._service_speech(friendly, action),
            "result": result,
        }

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
        if not self.has_entity_scope(cfg):
            return False
        normalized = _normalize_tr(text)
        weather_terms = ["hava", "derece", "sicaklik", "nem", "ruzgar", "yagmur"]
        device_terms = ["isik", "lamba", "priz", "klima", "perde", "panjur", "isitici", "fan", "sensor", "kamera"]
        action_terms = ["ac", "kapat", "yak", "sondur", "calistir", "durdur", "ayarla", "durum", "kac"]
        return any(term in normalized for term in weather_terms) or (
            any(term in normalized for term in device_terms) and any(term in normalized for term in action_terms)
        )

    async def _select_entity(self, text: str, domain_hint: str = "") -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        states = await self.list_states(domain=domain_hint, limit=256) if domain_hint else await self.list_states(limit=256)
        if not states and domain_hint:
            states = await self.list_states(limit=256)
        if not states:
            return None, []

        text_norm = _normalize_tr(text)
        terms = [term for term in _words(text) if len(term) > 1 and term not in _IGNORED_MATCH_TERMS]
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in states:
            entity_id = str(item.get("entity_id") or "").lower()
            friendly_name = str(item.get("friendly_name") or "")
            friendly_norm = _normalize_tr(friendly_name)
            haystack = f"{entity_id} {friendly_norm}"
            score = 0
            if entity_id and entity_id in text_norm:
                score += 40
            if domain_hint and entity_id.startswith(f"{domain_hint}."):
                score += 5
            for term in terms:
                if term in friendly_norm:
                    score += 8
                elif term in haystack:
                    score += 4
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("entity_id") or "")))
        if not scored:
            if domain_hint and len(states) == 1:
                return states[0], []
            return None, states[:5]
        if len(scored) > 1 and scored[0][0] == scored[1][0] and scored[0][0] < 14:
            return None, [item for _score, item in scored[:5]]
        return scored[0][1], [item for _score, item in scored[1:6]]

    def _detect_action(self, text: str) -> str:
        words = set(_words(text))
        if words & _TURN_OFF_TERMS:
            return "turn_off"
        if words & _TURN_ON_TERMS:
            return "turn_on"
        if words & _TOGGLE_TERMS:
            return "toggle"
        if words & _READ_TERMS:
            return "read"
        return ""

    def _domain_hint(self, text: str) -> str:
        for word in _words(text):
            for term, domain in _DOMAIN_HINTS.items():
                if word == term or word.startswith(term):
                    return domain
        return ""

    def _service_for_action(self, domain: str, action: str) -> str:
        if action == "toggle" and domain in {"light", "switch", "fan", "input_boolean"}:
            return "toggle"
        if action == "turn_on":
            if domain == "cover":
                return "open_cover"
            if domain == "lock":
                return "unlock"
            if domain in _CONTROL_DOMAINS:
                return "turn_on"
        if action == "turn_off":
            if domain == "cover":
                return "close_cover"
            if domain == "lock":
                return "lock"
            if domain in _CONTROL_DOMAINS:
                return "turn_off"
        return ""

    def _service_speech(self, friendly_name: str, action: str) -> str:
        if action == "turn_on":
            return f"{friendly_name} acildi."
        if action == "turn_off":
            return f"{friendly_name} kapatildi."
        if action == "toggle":
            return f"{friendly_name} degistirildi."
        return f"{friendly_name} icin komut uygulandi."

    def _state_speech(self, state: dict[str, Any], friendly_name: str) -> str:
        value = str(state.get("state") or "bilinmiyor")
        attributes = state.get("attributes") if isinstance(state.get("attributes"), dict) else {}
        unit = str(attributes.get("unit_of_measurement") or "").strip()
        if value in {"unknown", "unavailable"}:
            return f"{friendly_name} durumu su anda bilinmiyor."
        return f"{friendly_name}: {value}{(' ' + unit) if unit else ''}."

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
