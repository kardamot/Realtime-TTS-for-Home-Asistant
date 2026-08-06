from __future__ import annotations

import pathlib
import sys
import unittest


ADDON_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_ROOT))

from app.pipeline.tts.google_ai_request import (  # noqa: E402
    build_google_ai_interactions_payload,
    build_google_ai_tts_input,
)


class GoogleAiTtsRequestTests(unittest.TestCase):
    def test_plain_text_is_unchanged_without_style_instruction(self) -> None:
        text = "Merhaba Mustafa, bugün nasılsın?"

        self.assertEqual(build_google_ai_tts_input(text, "  "), text)

    def test_style_instruction_is_embedded_before_labeled_transcript(self) -> None:
        style = "Kadınsı, muzip ve hafif iğneleyici bir tonda konuş."
        text = "Devrelerim yerinde, senin keyfin nasıl?"

        prompt = build_google_ai_tts_input(text, style)

        self.assertIn(style, prompt)
        self.assertIn(text, prompt)
        self.assertLess(prompt.index(style), prompt.index(text))
        self.assertIn("<yonetmen_notlari>", prompt)
        self.assertIn("<seslendirilecek_metin>", prompt)
        self.assertIn("yalnızca <seslendirilecek_metin>", prompt.casefold())

    def test_interactions_payload_never_uses_developer_instruction(self) -> None:
        payload = build_google_ai_interactions_payload(
            model="gemini-3.1-flash-tts-preview",
            voice_name="Kore",
            text="Hazırım.",
            style_instruction="Neşeli ve karakterli konuş.",
        )

        self.assertNotIn("system_instruction", payload)
        self.assertNotIn("systemInstruction", payload)
        self.assertEqual(payload["model"], "gemini-3.1-flash-tts-preview")
        self.assertEqual(payload["response_format"], {"type": "audio"})
        self.assertEqual(payload["generation_config"]["speech_config"], [{"voice": "Kore"}])
        self.assertTrue(payload["stream"])
        self.assertIn("Neşeli ve karakterli konuş.", payload["input"])
        self.assertIn("Hazırım.", payload["input"])


if __name__ == "__main__":
    unittest.main()
