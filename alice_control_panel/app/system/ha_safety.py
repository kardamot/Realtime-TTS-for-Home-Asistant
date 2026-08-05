from __future__ import annotations

import re


TECHNICAL_HA_FALLBACK = "Ev kontrolü yanıtını güvenli biçimde tamamlayamadım. Cihazı tekrar söyler misin?"
_TECHNICAL_HA_OUTPUT_RE = re.compile(
    r"(?:"
    r"<\s*alice_control_panel\b"
    r"|homeassistant[._-]*service_call"
    r"|\b(?:light|switch|fan|climate|media_player|humidifier)\."
    r"(?:turn_on|turn_off|toggle|set_temperature|set_hvac_mode|volume_set)\b"
    r"|[\"']action[\"']\s*:\s*[\"']homeassistant\."
    r")",
    re.IGNORECASE,
)
_TECHNICAL_HA_JSON_DOMAIN_RE = re.compile(r"[\"']domain[\"']\s*:\s*[\"'][a-z_]+[\"']", re.IGNORECASE)
_TECHNICAL_HA_JSON_SERVICE_RE = re.compile(r"[\"']service[\"']\s*:\s*[\"'][a-z_]+[\"']", re.IGNORECASE)


def sanitize_assistant_output(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    technical_json = _TECHNICAL_HA_JSON_DOMAIN_RE.search(clean) and _TECHNICAL_HA_JSON_SERVICE_RE.search(clean)
    if _TECHNICAL_HA_OUTPUT_RE.search(clean) or technical_json:
        return TECHNICAL_HA_FALLBACK
    return clean
