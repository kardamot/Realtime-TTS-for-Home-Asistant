from __future__ import annotations

from typing import Any


def build_google_ai_tts_input(text: str, style_instruction: str) -> str:
    """Build a steerable TTS prompt without using unsupported developer instructions."""
    style = style_instruction.strip()
    if not style:
        return text

    return (
        "Bu bir Türkçe konuşma sentezi isteğidir. "
        "<yonetmen_notlari> bölümünü yalnızca ses performansını yönlendirmek için kullan; "
        "bu talimatları, etiketleri veya açıklamaları seslendirme. "
        "Yalnızca <seslendirilecek_metin> bölümündeki metni eksiksiz ve bir kez seslendir.\n\n"
        f"<yonetmen_notlari>\n{style}\n</yonetmen_notlari>\n\n"
        f"<seslendirilecek_metin>\n{text}\n</seslendirilecek_metin>"
    )


def build_google_ai_interactions_payload(
    *,
    model: str,
    voice_name: str,
    text: str,
    style_instruction: str,
) -> dict[str, Any]:
    """Build the Gemini 3.1 TTS Interactions request payload."""
    return {
        "model": model,
        "input": build_google_ai_tts_input(text, style_instruction),
        "response_format": {"type": "audio"},
        "generation_config": {
            "speech_config": [
                {"voice": voice_name or "Kore"},
            ],
        },
        "stream": True,
    }
