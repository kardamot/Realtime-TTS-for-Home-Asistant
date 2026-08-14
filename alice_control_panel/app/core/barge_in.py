from __future__ import annotations

from typing import Any


def realtime_turn_detection(realtime: dict[str, Any], barge_in_enabled: bool = True) -> dict[str, Any]:
    mode = str(realtime.get("turn_detection") or "server_vad").strip().lower()
    if mode == "semantic_vad":
        eagerness = str(realtime.get("semantic_eagerness") or "high").strip().lower()
        if eagerness not in {"low", "medium", "high", "auto"}:
            eagerness = "high"
        return {
            "type": "semantic_vad",
            "eagerness": eagerness,
            "create_response": False,
            "interrupt_response": bool(barge_in_enabled),
        }
    return {
        "type": "server_vad",
        "threshold": max(0.0, min(1.0, float(realtime.get("vad_threshold") or 0.5))),
        "prefix_padding_ms": max(0, int(realtime.get("prefix_padding_ms") or 300)),
        "silence_duration_ms": max(120, int(realtime.get("silence_duration_ms") or 420)),
        "create_response": False,
        "interrupt_response": bool(barge_in_enabled),
    }


async def sync_barge_in_to_esp(esp_client: Any, log_bus: Any, config: dict[str, Any]) -> dict[str, Any]:
    pipeline = config.get("pipeline") if isinstance(config.get("pipeline"), dict) else {}
    barge_lab = config.get("barge_lab") if isinstance(config.get("barge_lab"), dict) else {}
    enabled = bool(pipeline.get("barge_in_enabled", True))
    try:
        result = await esp_client.update_config({"barge_in_enabled": enabled, "barge_lab": barge_lab})
    except Exception as exc:
        result = {"ok": False, "message": str(exc)}
    await log_bus.emit(
        "INFO" if result.get("ok") else "WARN",
        "ESP",
        "Barge-in setting synchronized to ESP" if result.get("ok") else "Barge-in setting could not be synchronized to ESP",
        {"barge_in_enabled": enabled, "barge_lab_mode": barge_lab.get("mode"), "result": result},
    )
    return result
