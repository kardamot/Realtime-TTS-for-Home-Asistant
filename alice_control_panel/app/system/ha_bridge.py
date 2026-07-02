from __future__ import annotations

import fnmatch
import math
import os
import random
import re
from dataclasses import dataclass, field
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
_READ_TERMS = {
    "durum",
    "durumu",
    "kac",
    "nedir",
    "nasil",
    "oku",
    "goster",
    "sicaklik",
    "nem",
    "hava",
    "acik",
    "kapali",
    "yaniyor",
}
_ACTION_SUFFIXES = (
    "ar",
    "er",
    "ir",
    "ur",
    "iyor",
    "acak",
    "ecek",
    "abilir",
    "ebilir",
    "sana",
    "iver",
    "in",
    "alim",
)
_ENTITY_SUFFIXES = (
    "lerimizi",
    "larimizi",
    "lerini",
    "larini",
    "lerimi",
    "larimi",
    "nizi",
    "imizi",
    "sini",
    "sunu",
    "sine",
    "lara",
    "lere",
    "lar",
    "ler",
    "ni",
    "nu",
    "ne",
    "yi",
    "yu",
    "ye",
    "i",
    "u",
)
_DOMAIN_HINTS = {
    "isik": "light",
    "isig": "light",
    "isiklar": "light",
    "lamba": "light",
    "lamb": "light",
    "avize": "light",
    "led": "light",
    "renk": "light",
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
    "ses": "media_player",
    "muzik": "media_player",
    "hoparlor": "media_player",
    "televizyon": "media_player",
    "tv": "media_player",
    "sensor": "sensor",
    "sicaklik": "sensor",
    "sicak": "sensor",
    "derece": "sensor",
    "nem": "sensor",
    "hava": "weather",
    "kilit": "lock",
}
_TARGETABLE_DOMAIN_TERMS = {"avize", "led", "priz", "role", "fan", "vantilator", "perde", "panjur", "klima", "termostat", "tv"}
_IGNORED_MATCH_TERMS = (
    _TURN_ON_TERMS
    | _TURN_OFF_TERMS
    | _TOGGLE_TERMS
    | _READ_TERMS
    | set(_DOMAIN_HINTS)
    | {
        "alice",
        "lutfen",
        "bir",
        "de",
        "da",
        "mi",
        "mu",
        "misin",
        "musun",
        "midir",
        "bana",
        "icin",
        "su",
        "sunu",
        "oradaki",
        "yap",
        "al",
        "ayarla",
        "getir",
        "et",
        "eder",
        "ederim",
        "lazim",
    }
)

_ALL_TERMS = {
    "tum",
    "tumu",
    "tumunu",
    "butun",
    "hepsi",
    "hepsini",
    "tamami",
    "tamamini",
    "toplu",
}
_ONLY_TERMS = {"sadece", "yalniz", "yalnizca"}
_ROOM_TERMS = {
    "salon",
    "oturma",
    "oda",
    "mutfak",
    "yatak",
    "banyo",
    "koridor",
    "hol",
    "calisma",
    "cocuk",
    "misafir",
    "teras",
    "balkon",
    "bahce",
    "garaj",
}
_COLOR_RGB: dict[str, tuple[str, tuple[int, int, int]]] = {
    "kirmizi": ("kirmizi", (255, 0, 0)),
    "mavi": ("mavi", (0, 90, 255)),
    "yesil": ("yesil", (0, 180, 80)),
    "sari": ("sari", (255, 210, 0)),
    "turuncu": ("turuncu", (255, 120, 0)),
    "mor": ("mor", (145, 70, 255)),
    "pembe": ("pembe", (255, 90, 170)),
    "beyaz": ("beyaz", (255, 255, 255)),
}
_COLOR_TEMPERATURES: dict[str, tuple[str, int]] = {
    "sicak beyaz": ("sicak beyaz", 2700),
    "soguk beyaz": ("soguk beyaz", 6500),
    "gun isigi": ("gun isigi", 4000),
    "gunisigi": ("gun isigi", 4000),
    "normal": ("normal", 4000),
    "normale": ("normal", 4000),
}
_COLOR_WORDS = set(_COLOR_RGB) | {word for phrase in _COLOR_TEMPERATURES for word in phrase.split()} | {"renk", "renkleri"}
_BRIGHTNESS_WORDS = {
    "yuzde",
    "parlaklik",
    "parlakligi",
    "kis",
    "kismak",
    "azalt",
    "artir",
    "yukselt",
    "cogalt",
    "biraz",
    "los",
    "yari",
    "yariya",
    "yarim",
    "full",
    "tam",
    "son",
}
_RNG = random.SystemRandom()
_CLEAR_WEATHER_ADVICE = (
    "Dışarı çıkacaksan fena görünmüyor.",
    "Hava tarafında şimdilik sakin bir tablo var.",
    "Dışarı planı için kötü bir işaret görmüyorum.",
    "Bugün hava işi fazla naz yapmıyor.",
)


@dataclass
class HaIntent:
    action: str = ""
    domain_hint: str = ""
    target_terms: list[str] = field(default_factory=list)
    area_terms: list[str] = field(default_factory=list)
    color_name: str = ""
    rgb_color: tuple[int, int, int] | None = None
    color_temp_kelvin: int | None = None
    brightness_pct: int | None = None
    brightness_step_pct: int | None = None
    temperature: float | None = None
    hvac_mode: str = ""
    all_requested: bool = False
    only_requested: bool = False
    room_group_requested: bool = False


@dataclass
class EntityScore:
    score: int
    item: dict[str, Any]
    reasons: list[str] = field(default_factory=list)

_WEATHER_CONDITION_TR = {
    "clear-night": "a\u00e7\u0131k bir gece",
    "cloudy": "bulutlu",
    "fog": "sisli",
    "hail": "dolu riski olan",
    "lightning": "g\u00f6k g\u00fcr\u00fclt\u00fcl\u00fc",
    "lightning-rainy": "g\u00f6k g\u00fcr\u00fclt\u00fcl\u00fc ve ya\u011fmurlu",
    "partlycloudy": "par\u00e7al\u0131 bulutlu",
    "pouring": "sa\u011fanak ya\u011fmurlu",
    "rainy": "ya\u011fmurlu",
    "snowy": "karl\u0131",
    "snowy-rainy": "karla kar\u0131\u015f\u0131k ya\u011fmurlu",
    "sunny": "g\u00fcne\u015fli",
    "windy": "r\u00fczgarl\u0131",
    "windy-variant": "r\u00fczgarl\u0131 ve bulutlu",
    "exceptional": "ola\u011fan d\u0131\u015f\u0131",
}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}".replace(".", ",")


def _rounded_weather_int(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _is_nearly_integer(value: float, rounded: int) -> bool:
    return abs(value - rounded) < 0.15


def _format_temperature_phrase(value: float | None) -> str:
    if value is None:
        return ""
    rounded = _rounded_weather_int(value)
    if _is_nearly_integer(value, rounded):
        return f"{rounded} derece"
    return f"{rounded} derece civar\u0131"


def _format_weather_measure_phrase(value: float | None, unit: str = "") -> str:
    if value is None:
        return ""
    rounded = _rounded_weather_int(value)
    normalized_unit = unit.strip().lower()
    unit_text = "kilometre/saat" if normalized_unit in {"km/h", "kmh", "kph"} else unit.strip()
    suffix = "" if _is_nearly_integer(value, rounded) else " civar\u0131"
    return f"{rounded}{(' ' + unit_text) if unit_text else ''}{suffix}"


def _format_weather_percent_phrase(value: float | None) -> str:
    if value is None:
        return ""
    rounded = _rounded_weather_int(value)
    suffix = "" if _is_nearly_integer(value, rounded) else " civar\u0131"
    return f"y\u00fczde {rounded}{suffix}"


def _weather_condition_text(value: Any) -> str:
    key = str(value or "").strip().lower().replace("_", "-")
    return _WEATHER_CONDITION_TR.get(key, key or "bilinmiyor")


def _weather_query_scope(text: str) -> str:
    normalized = _normalize_tr(text)
    if any(term in normalized for term in ("yarin", "ertesi gun")):
        return "tomorrow"
    if any(term in normalized for term in ("bugun", "simdi", "su an", "disari", "disarida")):
        return "today"
    return "current"


def _forecast_response_data(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    service_response = response.get("service_response")
    return service_response if isinstance(service_response, dict) else response


def _forecast_items(state: dict[str, Any], forecast_type: str) -> list[dict[str, Any]]:
    responses = state.get("alice_forecast_response") if isinstance(state.get("alice_forecast_response"), dict) else {}
    preferred = [key for key in responses if key.endswith(f"_{forecast_type}")]
    fallback = [key for key in responses if key not in preferred]
    for key in [*preferred, *fallback]:
        response_data = _forecast_response_data(responses.get(key))
        for forecast_doc in response_data.values():
            if not isinstance(forecast_doc, dict):
                continue
            items = forecast_doc.get("forecast")
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _forecast_for_scope(state: dict[str, Any], user_text: str) -> tuple[str, dict[str, Any] | None]:
    scope = _weather_query_scope(user_text)
    if scope == "tomorrow":
        daily = _forecast_items(state, "daily")
        if len(daily) > 1:
            return "tomorrow", daily[1]
        if daily:
            return "tomorrow", daily[0]
    if scope in {"today", "current"}:
        hourly = _forecast_items(state, "hourly")
        if hourly:
            return "today", hourly[0]
        daily = _forecast_items(state, "daily")
        if daily:
            return "today", daily[0]
    return scope, None


def _weather_doc_number(doc: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(doc.get(key))
        if value is not None:
            return value
    return None


def _weather_advice(condition_key: str, temperature: float | None, wind_speed: float | None, precipitation: float | None) -> str:
    condition = condition_key.lower().replace("_", "-")
    advice: list[str] = []
    if condition in {"rainy", "pouring", "lightning-rainy"} or (precipitation is not None and precipitation > 0):
        advice.append("\u015eemsiye al; sonra dramatik dramatik ko\u015fmak zorunda kalma.")
    if condition in {"snowy", "snowy-rainy"}:
        advice.append("Kaygan zeminlere dikkat; kahramanl\u0131k yapma, Alice izliyor.")
    if wind_speed is not None:
        if wind_speed >= 50:
            advice.append("R\u00fczgar sert, sa\u00e7-ba\u015f modu biraz sava\u015f alan\u0131 olabilir.")
        elif wind_speed >= 30:
            advice.append("R\u00fczgar kendini belli ediyor, hafife alma.")
    if temperature is not None:
        if temperature <= 0:
            advice.append("S\u0131k\u0131 giyin; d\u0131\u015far\u0131s\u0131 \u0131s\u0131rma modunda.")
        elif temperature <= 8:
            advice.append("Kal\u0131n bir \u015fey almak iyi olur.")
        elif temperature <= 16:
            advice.append("\u0130nce bir ceket ak\u0131ll\u0131ca olur.")
        elif temperature >= 30:
            advice.append("Su i\u00e7meyi unutma; s\u0131cak taraflar hafif sinsice.")
    return advice[0] if advice else _RNG.choice(_CLEAR_WEATHER_ADVICE)


def _weather_location(friendly_name: str) -> str:
    return friendly_name.replace("Hava Durumu", "").replace("hava durumu", "").strip() or friendly_name


def _weather_speech(state: dict[str, Any], friendly_name: str, user_text: str = "") -> str:
    attributes = state.get("attributes") if isinstance(state.get("attributes"), dict) else {}
    current_state = str(state.get("state") or "bilinmiyor")
    scope, forecast = _forecast_for_scope(state, user_text)
    source = forecast if forecast else attributes
    condition_key = str((forecast or {}).get("condition") or current_state)
    condition = _weather_condition_text(condition_key)
    temperature = _weather_doc_number(source, "temperature", "templow", "temperature_low")
    high_temperature = _weather_doc_number(source, "temperature", "temperature_high")
    low_temperature = _weather_doc_number(source, "templow", "temperature_low", "low_temperature")
    humidity = _weather_doc_number(source, "humidity")
    wind_speed = _weather_doc_number(source, "wind_speed", "wind_speed_10m")
    precipitation_probability = _weather_doc_number(source, "precipitation_probability")
    precipitation_amount = _weather_doc_number(source, "precipitation")
    precipitation = precipitation_probability if precipitation_probability is not None else precipitation_amount
    wind_unit = str(source.get("wind_speed_unit") or attributes.get("wind_speed_unit") or "km/h").strip()
    location = _weather_location(friendly_name)
    when = "Yar\u0131n" if scope == "tomorrow" and forecast else "\u015eu an"

    pieces = [f"{when} {location} i\u00e7in hava {condition}"]
    if high_temperature is not None and low_temperature is not None and abs(high_temperature - low_temperature) > 0.2:
        low_text = str(_rounded_weather_int(low_temperature))
        high_text = str(_rounded_weather_int(high_temperature))
        pieces.append(f"s\u0131cakl\u0131k {low_text} ile {high_text} derece aras\u0131")
    elif temperature is not None:
        pieces.append(f"s\u0131cakl\u0131k {_format_temperature_phrase(temperature)}")
    if wind_speed is not None:
        pieces.append(f"r\u00fczgar {_format_weather_measure_phrase(wind_speed, wind_unit)}")
    if humidity is not None and scope != "tomorrow":
        pieces.append(f"nem {_format_weather_percent_phrase(humidity)}")
    if precipitation_probability is not None and precipitation_probability > 0:
        pieces.append(f"ya\u011f\u0131\u015f olas\u0131l\u0131\u011f\u0131 {_format_weather_percent_phrase(precipitation_probability)}")
    elif precipitation_amount is not None and precipitation_amount > 0:
        pieces.append(f"ya\u011f\u0131\u015f {_format_weather_measure_phrase(precipitation_amount, 'mm')}")

    advice = _weather_advice(condition_key, temperature or high_temperature, wind_speed, precipitation)
    return ", ".join(pieces) + f". {advice}"


_CONTROL_DOMAINS = {"light", "switch", "fan", "input_boolean", "media_player", "climate", "humidifier"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_.]+", _normalize_tr(text))


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_tr(text))


def _clamp_pct(value: int) -> int:
    return max(1, min(100, int(value)))


def _phrase_in_text(phrase: str, text_norm: str, text_compact: str) -> bool:
    phrase_norm = _normalize_tr(phrase).strip()
    if not phrase_norm:
        return False
    return phrase_norm in text_norm or _compact_text(phrase_norm) in text_compact


def _matches_action_word(word: str, roots: set[str]) -> bool:
    if word in roots:
        return True
    for root in roots:
        if not word.startswith(root):
            continue
        suffix = word[len(root) :]
        if suffix and suffix.startswith(_ACTION_SUFFIXES):
            return True
    return False


def _is_ignored_entity_term(term: str) -> bool:
    return (
        (term in _IGNORED_MATCH_TERMS and term not in _TARGETABLE_DOMAIN_TERMS)
        or _matches_action_word(term, _TURN_ON_TERMS)
        or _matches_action_word(term, _TURN_OFF_TERMS)
        or _matches_action_word(term, _TOGGLE_TERMS)
    )


def _entity_match_terms(text: str) -> list[str]:
    terms: list[str] = []
    for word in _words(text):
        if len(word) <= 1 or _is_ignored_entity_term(word):
            continue
        candidates = [word]
        for suffix in _ENTITY_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                candidates.append(word[: -len(suffix)])
        for candidate in candidates:
            if len(candidate) > 1 and candidate not in terms and not _is_ignored_entity_term(candidate):
                terms.append(candidate)
    return terms


def _alias_map(cfg: dict[str, Any]) -> dict[str, list[str]]:
    raw = cfg.get("aliases")
    aliases: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for entity_id, value in raw.items():
            key = str(entity_id or "").strip().lower()
            if not key:
                continue
            if isinstance(value, str):
                items = re.split(r"[,;\n]+", value)
            elif isinstance(value, list):
                items = [str(item) for item in value]
            else:
                items = []
            clean = [item.strip() for item in items if item and item.strip()]
            if clean:
                aliases[key] = clean
        return aliases
    if not isinstance(raw, str):
        return aliases
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        entity_id, values = line.split(":", 1)
        key = entity_id.strip().lower()
        if not key:
            continue
        clean = [item.strip() for item in re.split(r"[,;]+", values) if item.strip()]
        if clean:
            aliases[key] = clean
    return aliases


def _friendly_name(item: dict[str, Any]) -> str:
    return str(item.get("friendly_name") or item.get("entity_id") or "")


def _sentence_name(value: str) -> str:
    clean = re.sub(r"\s+", " ", str(value or "")).strip()
    if not clean:
        return ""
    return clean[0].upper() + clean[1:]


def _display_list(items: list[dict[str, Any]], limit: int = 3) -> str:
    names: list[str] = []
    for item in items[:limit]:
        name = _friendly_name(item)
        if name and name not in names:
            names.append(name)
    return ", ".join(names)


def _split_target_terms(text: str) -> tuple[list[str], list[str]]:
    target_terms = _entity_match_terms(text)
    filtered: list[str] = []
    area_terms: list[str] = []
    for term in target_terms:
        if term.isdigit():
            continue
        if term in _ALL_TERMS or term in _ONLY_TERMS or term in _COLOR_WORDS or term in _BRIGHTNESS_WORDS:
            continue
        if any(color.startswith(term) or term.startswith(color) for color in _COLOR_RGB):
            continue
        if term.startswith(("isik", "isig", "lamba", "lamb")):
            continue
        if term.startswith("oda"):
            if "oda" not in area_terms:
                area_terms.append("oda")
            if term != "oda":
                continue
        if term in {"yap", "al", "ayarla", "normal", "normale"}:
            continue
        if term not in filtered:
            filtered.append(term)
        if term in _ROOM_TERMS and term not in area_terms:
            area_terms.append(term)
    return filtered, area_terms


class HomeAssistantBridge:
    def __init__(self, config_store: ConfigStore, log_bus: LogBus) -> None:
        self._config_store = config_store
        self._log_bus = log_bus

    def parse_intent(self, text: str) -> HaIntent:
        text_norm = _normalize_tr(text)
        text_compact = _compact_text(text_norm)
        words = _words(text)
        intent = HaIntent()
        intent.domain_hint = self._domain_hint(text)
        intent.only_requested = any(word in _ONLY_TERMS for word in words)
        intent.all_requested = self._all_requested(words)

        color_name, rgb_color, color_temp_kelvin = self._parse_color(text_norm, text_compact)
        if color_name:
            intent.action = "set_color"
            intent.color_name = color_name
            intent.rgb_color = rgb_color
            intent.color_temp_kelvin = color_temp_kelvin
            if not intent.domain_hint or intent.domain_hint in {"sensor", "weather"}:
                intent.domain_hint = "light"
            if any(word.startswith("renkler") for word in words):
                intent.all_requested = True

        brightness_pct, brightness_step_pct = self._parse_brightness(text_norm, words)
        if brightness_pct is not None or brightness_step_pct is not None:
            if intent.domain_hint == "media_player" or "ses" in words:
                intent.action = "set_media_volume"
            elif not intent.action:
                intent.action = "set_brightness"
            intent.brightness_pct = brightness_pct
            intent.brightness_step_pct = brightness_step_pct
            if not intent.domain_hint:
                intent.domain_hint = "light"

        temperature = self._parse_temperature(text_norm, words)
        if temperature is not None and intent.domain_hint == "climate":
            intent.action = "set_temperature"
            intent.temperature = temperature

        hvac_mode = self._parse_hvac_mode(words)
        if hvac_mode and intent.domain_hint == "climate":
            intent.action = "set_hvac"
            intent.hvac_mode = hvac_mode

        if not intent.action:
            intent.action = self._detect_action(text)

        intent.target_terms, intent.area_terms = _split_target_terms(text)
        specific_terms = [term for term in intent.target_terms if term not in intent.area_terms]
        intent.room_group_requested = bool(
            intent.domain_hint == "light"
            and intent.area_terms
            and not specific_terms
            and not intent.only_requested
        )
        return intent

    def _parse_color(self, text_norm: str, text_compact: str) -> tuple[str, tuple[int, int, int] | None, int | None]:
        for phrase, (name, kelvin) in sorted(_COLOR_TEMPERATURES.items(), key=lambda item: len(item[0]), reverse=True):
            if _phrase_in_text(phrase, text_norm, text_compact):
                return name, None, kelvin
        words = set(_words(text_norm))
        for key, (name, rgb) in _COLOR_RGB.items():
            if key in words or _phrase_in_text(key, text_norm, text_compact):
                return name, rgb, None
        return "", None, None

    def _parse_brightness(self, text_norm: str, words: list[str]) -> tuple[int | None, int | None]:
        match = re.search(r"(?:yuzde|%)\s*(\d{1,3})", text_norm)
        if not match:
            match = re.search(r"\b(\d{1,3})\s*(?:yuzde|%)", text_norm)
        if match:
            return _clamp_pct(int(match.group(1))), None
        word_set = set(words)
        if word_set & {"yari", "yariya", "yarim"}:
            return 50, None
        if word_set & {"full", "ful", "tam", "son"}:
            return 100, None
        if "los" in word_set:
            return 20, None
        has_softener = "biraz" in word_set
        if any(word.startswith(("kis", "azalt", "dusur")) for word in words):
            return None, -15 if has_softener else -25
        if any(word.startswith(("artir", "yukselt", "cogalt")) for word in words):
            return None, 15 if has_softener else 25
        if has_softener and any(_matches_action_word(word, _TURN_ON_TERMS) for word in words):
            return None, 15
        return None, None

    def _parse_temperature(self, text_norm: str, words: list[str]) -> float | None:
        if not any(word in {"derece", "sicaklik", "klima", "termostat"} for word in words):
            return None
        match = re.search(r"\b(\d{1,2}(?:[,.]\d)?)\s*(?:derece|c)\b", text_norm)
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None

    def _parse_hvac_mode(self, words: list[str]) -> str:
        word_set = set(words)
        if word_set & {"isit", "isitma", "sicak"}:
            return "heat"
        if word_set & {"sogut", "sogutma", "serinlet"}:
            return "cool"
        if word_set & {"oto", "otomatik", "auto"}:
            return "auto"
        return ""

    def _all_requested(self, words: list[str]) -> bool:
        if any(word in _ALL_TERMS for word in words):
            return True
        return any(word.startswith(("isiklar", "isiklari", "lambalar", "lambalari")) for word in words)

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
                "alias_count": len(_alias_map(cfg)),
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
                "alias_count": len(_alias_map(cfg)),
                "entity_scope": self.has_entity_scope(cfg),
            }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self._base_url(cfg)}/config", headers=self._headers(token)) as resp:
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
                        "alias_count": len(_alias_map(cfg)),
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
                "alias_count": len(_alias_map(cfg)),
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

    async def get_weather_forecast_response(self, entity_id: str) -> dict[str, Any]:
        cfg = await self._cfg()
        self.assert_entity_allowed(entity_id, cfg)
        responses: dict[str, Any] = {}
        for service_name in ("get_forecasts", "get_forecast"):
            for forecast_type in ("hourly", "daily"):
                try:
                    async with self._session() as session:
                        async with session.post(
                            f"{self._base_url(cfg)}/services/weather/{service_name}?return_response",
                            headers=self._headers(),
                            json={"entity_id": entity_id, "type": forecast_type},
                        ) as resp:
                            if resp.status >= 400:
                                continue
                            responses[f"{service_name}_{forecast_type}"] = await resp.json()
                except Exception:
                    continue
            if responses:
                break
        return responses

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
        cfg = await self._cfg()
        query_terms = _entity_match_terms(query)
        if not query_terms:
            return []
        states = await self.list_states(domain=domain, limit=256)
        aliases = _alias_map(cfg)
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in states:
            entity_id = str(item.get("entity_id") or "")
            friendly_name = str(item.get("friendly_name") or "")
            haystack = _normalize_tr(f"{entity_id} {friendly_name} {' '.join(aliases.get(entity_id.lower(), []))}")
            compact = _compact_text(haystack)
            score = 0
            for term in query_terms:
                if term in haystack:
                    score += 3
                if term in compact:
                    score += 2
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

    async def handle_text_command(self, text: str) -> dict[str, Any]:
        cfg = await self._cfg()
        intent = self.parse_intent(text)
        if not intent.action:
            return {"handled": False, "ok": False, "reason": "no_home_assistant_intent"}
        if not bool(cfg.get("enabled", True)):
            return {"handled": True, "ok": False, "speech": "Home Assistant baglantisi kapali."}
        if not bool(cfg.get("route_home_control", True)):
            return {"handled": False, "ok": False, "reason": "routing_disabled"}
        if not self.has_entity_scope(cfg):
            return {"handled": True, "ok": False, "speech": "Home Assistant allowlist bos. Once izin verilen entity listesini doldurmalisin."}
        if not await self.is_ready():
            return {"handled": True, "ok": False, "speech": "Home Assistant API hazir degil."}

        entities, alternatives, clarification = await self._select_entities_for_intent(text, intent, cfg)
        if clarification:
            return {
                "handled": True,
                "ok": False,
                "speech": clarification,
                "requires_clarification": True,
            }
        if not entities:
            if intent.domain_hint:
                suffix = f" Adaylar: {_display_list(alternatives)}." if alternatives else ""
                return {"handled": True, "ok": False, "speech": f"Allowlist icinde uygun cihaz bulamadim.{suffix}"}
            return {"handled": False, "ok": False, "reason": "no_matching_entity"}

        domains = {str(item.get("entity_id") or "").split(".", 1)[0] for item in entities if "." in str(item.get("entity_id") or "")}
        if len(domains) != 1:
            return {
                "handled": True,
                "ok": False,
                "speech": "Bu komut icin tek bir cihaz turu secmem gerekiyor. Hangisini istedigini soyler misin?",
                "requires_clarification": True,
            }
        domain = next(iter(domains))
        friendly = self._friendly_selection_name(entities, intent, cfg)
        entity_ids = [str(item.get("entity_id") or "") for item in entities if str(item.get("entity_id") or "")]

        if not self._domain_supports_intent(domain, intent):
            return {
                "handled": True,
                "ok": False,
                "entity_id": entity_ids[0] if entity_ids else "",
                "domain": domain,
                "speech": self._domain_mismatch_speech(friendly, domain, intent),
            }

        if intent.action == "read":
            if len(entities) > 1:
                speech = await self._multi_state_speech(entities, text)
                await self._log_bus.emit("INFO", "HA", "Allowlisted HA state group read", {"count": len(entities), "domain": domain})
                return {
                    "handled": True,
                    "ok": True,
                    "action": intent.action,
                    "domain": domain,
                    "speech": speech,
                    "entity_ids": entity_ids,
                }
            entity_id = entity_ids[0]
            state = await self.get_state(entity_id)
            state_doc = state or entities[0]
            if domain == "weather":
                forecast_response = await self.get_weather_forecast_response(entity_id)
                if forecast_response:
                    state_doc = {**state_doc, "alice_forecast_response": forecast_response}
            speech = self._state_speech(state_doc, friendly, user_text=text)
            await self._log_bus.emit("INFO", "HA", "Allowlisted HA state read", {"entity_id": entity_id, "domain": domain})
            return {
                "handled": True,
                "ok": True,
                "action": intent.action,
                "entity_id": entity_id,
                "domain": domain,
                "friendly_name": friendly,
                "spoken_name": friendly,
                "speech": speech,
                "state": state_doc,
                "narration_kind": "weather" if entity_id.startswith("weather.") else "",
            }

        service, data = self._service_for_intent(domain, intent, entity_ids)
        if not service:
            return {
                "handled": True,
                "ok": False,
                "entity_id": entity_ids[0] if entity_ids else "",
                "domain": domain,
                "speech": f"{friendly} icin bu komut henuz desteklenmiyor.",
            }
        result = await self.call_service(domain, service, data)
        await self._log_bus.emit(
            "INFO",
            "HA",
            "Allowlisted HA service call",
            {"entity_ids": entity_ids, "domain": domain, "service": service, "action": intent.action},
        )
        return {
            "handled": True,
            "ok": True,
            "action": intent.action,
            "entity_id": entity_ids[0] if len(entity_ids) == 1 else "",
            "entity_ids": entity_ids,
            "domain": domain,
            "service": service,
            "friendly_name": friendly,
            "spoken_name": friendly,
            "speech": self._service_speech(friendly, intent, domain, len(entity_ids)),
            "narration_kind": "home_control",
            "result": result,
        }

    async def should_route_home_control(self, text: str) -> bool:
        cfg = await self._cfg()
        if not bool(cfg.get("route_home_control", True)):
            return False
        normalized = _normalize_tr(text)
        intent = self.parse_intent(text)
        if intent.action in {"set_color", "set_brightness", "set_temperature", "set_hvac", "set_media_volume"}:
            return True
        weather_terms = ["hava", "derece", "sicaklik", "nem", "ruzgar", "yagmur"]
        device_terms = ["isik", "lamba", "led", "renk", "priz", "klima", "perde", "panjur", "isitici", "fan", "sensor", "kamera", "ses", "muzik", "tv"]
        action_terms = ["ac", "kapat", "yak", "sondur", "calistir", "durdur", "ayarla", "durum", "kac", "yap", "kis", "artir", "azalt", "oku"]
        return any(term in normalized for term in weather_terms) or (
            bool(intent.action) and any(term in normalized for term in device_terms) and any(term in normalized for term in action_terms)
        )

    async def _select_entities_for_intent(
        self,
        text: str,
        intent: HaIntent,
        cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        domain_hint = intent.domain_hint
        states = await self.list_states(domain=domain_hint, limit=256) if domain_hint else await self.list_states(limit=256)
        if domain_hint and (not states or self._should_try_cross_domain_match(text, intent, states, cfg)):
            all_states = await self.list_states(limit=256)
            if all_states:
                states = all_states
        if not states:
            return [], [], ""

        if intent.all_requested and not domain_hint:
            domains = sorted({str(item.get("entity_id") or "").split(".", 1)[0] for item in states if "." in str(item.get("entity_id") or "")})
            names = ", ".join(self._domain_label(domain) for domain in domains[:3])
            return [], states[:5], f"Hepsini derken hangi grubu kastediyorsun: {names}?"

        scored = self._score_entities(text, intent, states, cfg)
        alternatives = [entry.item for entry in scored[:5]] if scored else states[:5]

        if intent.all_requested:
            if intent.target_terms or intent.area_terms:
                selected = [entry.item for entry in scored if entry.score >= 8]
                if not selected:
                    return [], alternatives, self._no_match_speech(intent, alternatives)
                return selected, alternatives, ""
            return states, alternatives, ""

        if intent.room_group_requested:
            selected = [entry.item for entry in scored if entry.score >= 8]
            if not selected:
                return [], alternatives, self._no_match_speech(intent, alternatives)
            return selected, alternatives, ""

        if not scored:
            if domain_hint and len(states) == 1:
                return [states[0]], [], ""
            return [], alternatives, self._clarify_speech(alternatives) if domain_hint and len(states) > 1 else ""

        top = scored[0]
        if top.score < 8:
            if domain_hint and len(states) == 1:
                return [states[0]], [], ""
            return [], alternatives, self._clarify_speech(alternatives)

        if len(scored) > 1:
            second = scored[1]
            close_scores = second.score >= max(8, top.score - 5)
            weak_top = top.score < 35
            if close_scores and weak_top:
                return [], [entry.item for entry in scored[:5]], self._clarify_speech([entry.item for entry in scored[:5]])

        return [top.item], [entry.item for entry in scored[1:6]], ""

    def _should_try_cross_domain_match(
        self,
        text: str,
        intent: HaIntent,
        states: list[dict[str, Any]],
        cfg: dict[str, Any],
    ) -> bool:
        if intent.all_requested or intent.room_group_requested:
            return False
        hinted_scores = self._score_entities(text, intent, states, cfg)
        hinted_top = hinted_scores[0].score if hinted_scores else 0
        if hinted_top >= 35:
            return False
        return bool(intent.target_terms)

    def _score_entities(self, text: str, intent: HaIntent, states: list[dict[str, Any]], cfg: dict[str, Any]) -> list[EntityScore]:
        text_norm = _normalize_tr(text)
        text_compact = _compact_text(text_norm)
        aliases = _alias_map(cfg)
        scored: list[EntityScore] = []
        for item in states:
            entity_id = str(item.get("entity_id") or "").lower()
            friendly_name = _friendly_name(item)
            friendly_norm = _normalize_tr(friendly_name)
            entity_words = _words(entity_id.replace(".", " ") + " " + friendly_name)
            alias_values = aliases.get(entity_id, [])
            alias_text = " ".join(alias_values)
            haystack = _normalize_tr(f"{entity_id} {friendly_name} {alias_text}")
            haystack_compact = _compact_text(haystack)
            score = 0
            reasons: list[str] = []
            if entity_id and entity_id in text_norm:
                score += 60
                reasons.append("entity_id")
            if friendly_norm and _compact_text(friendly_norm) and _compact_text(friendly_norm) in text_compact:
                score += 45
                reasons.append("friendly_phrase")
            for alias in alias_values:
                if _phrase_in_text(alias, text_norm, text_compact):
                    score += 50
                    reasons.append("alias_phrase")
            if intent.domain_hint and entity_id.startswith(f"{intent.domain_hint}."):
                score += 4
                reasons.append("domain")
            for term in intent.target_terms:
                if term in entity_words:
                    score += 12
                    reasons.append(f"word:{term}")
                elif term in haystack_compact:
                    score += 8
                    reasons.append(f"compact:{term}")
                elif term in haystack:
                    score += 5
                    reasons.append(f"contains:{term}")
            for term in intent.area_terms:
                if term in entity_words or term in haystack_compact:
                    score += 8
                    reasons.append(f"area:{term}")
            if score > 0:
                scored.append(EntityScore(score=score, item=item, reasons=reasons))
        scored.sort(key=lambda entry: (-entry.score, str(entry.item.get("entity_id") or "")))
        return scored

    def _clarify_speech(self, alternatives: list[dict[str, Any]]) -> str:
        names = _display_list(alternatives)
        if not names:
            return "Hangi cihazi kastettigini biraz daha net soyler misin?"
        return f"Birden fazla aday buldum: {names}. Hangisini istiyorsun?"

    def _no_match_speech(self, intent: HaIntent, alternatives: list[dict[str, Any]]) -> str:
        names = _display_list(alternatives)
        suffix = f" Adaylar: {names}." if names else ""
        if intent.area_terms:
            return f"Bu oda veya hedef icin allowlist icinde uygun cihaz bulamadim.{suffix}"
        return f"Allowlist icinde uygun cihaz bulamadim.{suffix}"

    def _detect_action(self, text: str) -> str:
        words = _words(text)
        if any(_matches_action_word(word, _TURN_OFF_TERMS) for word in words):
            return "turn_off"
        if any(_matches_action_word(word, _TURN_ON_TERMS) for word in words):
            return "turn_on"
        if any(_matches_action_word(word, _TOGGLE_TERMS) for word in words):
            return "toggle"
        if set(words) & _READ_TERMS:
            return "read"
        return ""

    def _domain_hint(self, text: str) -> str:
        for word in _words(text):
            for term, domain in _DOMAIN_HINTS.items():
                if word == term or word.startswith(term) or (len(term) >= 4 and term in word):
                    return domain
        return ""

    def _service_for_action(self, domain: str, action: str) -> str:
        if action == "toggle" and domain in {"light", "switch", "fan", "input_boolean", "media_player"}:
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

    def _service_for_intent(self, domain: str, intent: HaIntent, entity_ids: list[str]) -> tuple[str, dict[str, Any]]:
        entity_value: str | list[str] = entity_ids[0] if len(entity_ids) == 1 else entity_ids
        data: dict[str, Any] = {"entity_id": entity_value}
        if intent.action == "set_color":
            if domain != "light":
                return "", {}
            if intent.rgb_color is not None:
                data["rgb_color"] = list(intent.rgb_color)
            if intent.color_temp_kelvin is not None:
                data["color_temp_kelvin"] = intent.color_temp_kelvin
            return "turn_on", data
        if intent.action == "set_brightness":
            if domain != "light":
                return "", {}
            if intent.brightness_pct is not None:
                data["brightness_pct"] = intent.brightness_pct
            if intent.brightness_step_pct is not None:
                data["brightness_step_pct"] = intent.brightness_step_pct
            return "turn_on", data
        if intent.action == "set_temperature":
            if domain != "climate" or intent.temperature is None:
                return "", {}
            data["temperature"] = intent.temperature
            return "set_temperature", data
        if intent.action == "set_hvac":
            if domain != "climate" or not intent.hvac_mode:
                return "", {}
            data["hvac_mode"] = intent.hvac_mode
            return "set_hvac_mode", data
        if intent.action == "set_media_volume":
            if domain != "media_player" or intent.brightness_pct is None:
                return "", {}
            data["volume_level"] = round(intent.brightness_pct / 100, 2)
            return "volume_set", data
        service = self._service_for_action(domain, intent.action)
        return service, data if service else {}

    def _domain_supports_intent(self, domain: str, intent: HaIntent) -> bool:
        if intent.action in {"set_color", "set_brightness"}:
            return domain == "light"
        if intent.action in {"set_temperature", "set_hvac"}:
            return domain == "climate"
        if intent.action == "set_media_volume":
            return domain == "media_player"
        return True

    def _domain_mismatch_speech(self, friendly_name: str, domain: str, intent: HaIntent) -> str:
        if intent.action in {"set_color", "set_brightness"}:
            return f"Bunu yapamam; {friendly_name} bir isik degil."
        if intent.action in {"set_temperature", "set_hvac"}:
            return f"Bunu yapamam; {friendly_name} klima veya termostat degil."
        if intent.action == "set_media_volume":
            return f"Bunu yapamam; {friendly_name} medya oynatici degil."
        return f"{friendly_name} icin bu komut uygun degil."

    def _spoken_entity_name(self, entity: dict[str, Any], cfg: dict[str, Any]) -> str:
        entity_id = str(entity.get("entity_id") or "").lower()
        aliases = _alias_map(cfg).get(entity_id, [])
        if aliases:
            return _sentence_name(aliases[0])
        return _sentence_name(_friendly_name(entity))

    def _friendly_selection_name(self, entities: list[dict[str, Any]], intent: HaIntent, cfg: dict[str, Any]) -> str:
        if len(entities) == 1:
            return self._spoken_entity_name(entities[0], cfg)
        if intent.area_terms:
            area = "oturma odasi" if "oturma" in intent.area_terms and "oda" in intent.area_terms else " ".join(intent.area_terms)
            return _sentence_name(f"{area} isiklari")
        domain = str(entities[0].get("entity_id") or "").split(".", 1)[0]
        return _sentence_name(f"{len(entities)} {self._domain_label(domain)}")

    def _domain_label(self, domain: str) -> str:
        return {
            "light": "isik",
            "switch": "priz/anahtar",
            "fan": "fan",
            "climate": "klima",
            "media_player": "medya oynatici",
            "weather": "hava durumu",
            "sensor": "sensor",
            "cover": "perde/panjur",
            "lock": "kilit",
        }.get(domain, domain or "cihaz")

    def _service_speech(self, friendly_name: str, intent: HaIntent, domain: str, count: int = 1) -> str:
        target = friendly_name
        if intent.action == "turn_on":
            if domain == "light":
                return _RNG.choice((f"{target} açıldı.", f"{target} tamam, açtım.", f"{target} şimdi açık."))
            if domain == "switch":
                return _RNG.choice((f"{target} açıldı.", f"{target} devrede.", f"{target} tamam, açtım."))
            return _RNG.choice((f"{target} açıldı.", f"{target} tamam, çalışıyor."))
        if intent.action == "turn_off":
            return _RNG.choice((f"{target} kapatıldı.", f"{target} tamam, kapattım.", f"{target} artık kapalı."))
        if intent.action == "toggle":
            return _RNG.choice((f"{target} değiştirildi.", f"{target} durumunu çevirdim."))
        if intent.action == "set_color":
            color = intent.color_name or "istenen renge"
            return _RNG.choice((f"{target} {color} rengine alındı.", f"{target} için {color} tonu hazır.", f"{target} rengini {color} yaptım."))
        if intent.action == "set_brightness":
            if intent.brightness_pct is not None:
                return _RNG.choice(
                    (
                        f"{target} parlaklığı yüzde {intent.brightness_pct} oldu.",
                        f"{target} ışığını yüzde {intent.brightness_pct} yaptım.",
                        f"{target} yüzde {intent.brightness_pct} seviyesinde.",
                    )
                )
            if intent.brightness_step_pct is not None and intent.brightness_step_pct < 0:
                return _RNG.choice((f"{target} biraz kısıldı.", f"{target} daha loş oldu.", f"{target} ışığını biraz azalttım."))
            if intent.brightness_step_pct is not None:
                return _RNG.choice((f"{target} biraz açıldı.", f"{target} ışığını biraz yükselttim.", f"{target} daha parlak oldu."))
        if intent.action == "set_temperature" and intent.temperature is not None:
            value = int(intent.temperature) if intent.temperature.is_integer() else intent.temperature
            return _RNG.choice((f"{target} {value} dereceye ayarlandı.", f"{target} için {value} dereceyi seçtim."))
        if intent.action == "set_hvac":
            return _RNG.choice((f"{target} modu ayarlandı.", f"{target} modunu değiştirdim."))
        if intent.action == "set_media_volume" and intent.brightness_pct is not None:
            return _RNG.choice((f"{target} sesi yüzde {intent.brightness_pct} oldu.", f"{target} sesini yüzde {intent.brightness_pct} yaptım."))
        return _RNG.choice((f"{target} için komut uygulandı.", f"{target} tamam."))

    async def _multi_state_speech(self, entities: list[dict[str, Any]], user_text: str) -> str:
        pieces: list[str] = []
        for entity in entities[:5]:
            entity_id = str(entity.get("entity_id") or "")
            state = await self.get_state(entity_id)
            pieces.append(self._state_speech(state or entity, _friendly_name(entity), user_text=user_text).rstrip("."))
        if len(entities) > 5:
            pieces.append(f"ve {len(entities) - 5} cihaz daha")
        return ". ".join(piece for piece in pieces if piece) + "."

    def _state_speech(self, state: dict[str, Any], friendly_name: str, user_text: str = "") -> str:
        value = str(state.get("state") or "bilinmiyor")
        attributes = state.get("attributes") if isinstance(state.get("attributes"), dict) else {}
        unit = str(attributes.get("unit_of_measurement") or "").strip()
        entity_id = str(state.get("entity_id") or "")
        if value in {"unknown", "unavailable"}:
            return f"{friendly_name} durumu su anda bilinmiyor."
        if entity_id.startswith("weather."):
            return _weather_speech(state, friendly_name, user_text)
            temperature = _float_or_none(attributes.get("temperature"))
            temperature_unit = str(attributes.get("temperature_unit") or unit or "C").strip()
            humidity = _float_or_none(attributes.get("humidity"))
            wind_speed = _float_or_none(attributes.get("wind_speed"))
            wind_unit = str(attributes.get("wind_speed_unit") or "").strip()
            condition = _WEATHER_CONDITION_TR.get(value.lower().replace("_", "-"), value)
            location = friendly_name.replace("Hava Durumu", "").replace("hava durumu", "").strip() or friendly_name
            bits = [f"{location} tarafında hava {condition}"]
            if temperature is not None:
                bits.append(f"sıcaklık {_format_number(temperature)} derece")
            if wind_speed is not None:
                bits.append(f"rüzgar {_format_number(wind_speed)} {wind_unit or 'civarında'}")
            if humidity is not None:
                bits.append(f"nem yüzde {_format_number(humidity)}")

            advice: list[str] = []
            if value.lower() in {"rainy", "pouring", "lightning-rainy", "snowy-rainy"}:
                advice.append("Şemsiye fikri bugün fena değil")
            if value.lower() in {"snowy", "snowy-rainy"}:
                advice.append("Kaygan zeminlere dikkat; temkinli olmak iyi olur")
            if temperature is not None:
                if temperature <= 0:
                    advice.append("Sıkı giyin, dışarısı bayağı ısırıyor")
                elif temperature <= 8:
                    advice.append("Kalın bir şey almak iyi olur")
                elif temperature <= 16:
                    advice.append("İnce bir ceket iyi gider")
                elif temperature >= 28:
                    advice.append("Su içmeyi unutma, sıcak taraflara geçmişiz")
            if wind_speed is not None and wind_speed >= 20:
                advice.append("Rüzgar da kendini belli ediyor")

            sentence = ", ".join(bits) + "."
            if advice:
                sentence += " " + advice[0] + "."
            return sentence
        domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
        if domain in {"light", "switch", "fan", "input_boolean"} and value in {"on", "off"}:
            return f"{friendly_name} {'acik' if value == 'on' else 'kapali'}."
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
