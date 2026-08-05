from __future__ import annotations

import pathlib
import random
import sys
import types
import unittest
from typing import Any


ADDON_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_ROOT))

# The host-side parser tests do not make HTTP requests. Keep imports independent
# from add-on-only packages so they can run with a plain Python installation.
sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
config_store_module = types.ModuleType("app.core.config_store")
log_bus_module = types.ModuleType("app.core.log_bus")
config_store_module.ConfigStore = object
log_bus_module.LogBus = object
sys.modules.setdefault("app.core.config_store", config_store_module)
sys.modules.setdefault("app.core.log_bus", log_bus_module)

from app.system.ha_bridge import HomeAssistantBridge  # noqa: E402
from app.system.ha_response import HaResponseComposer  # noqa: E402
from app.system.ha_safety import sanitize_assistant_output  # noqa: E402


HA_CONFIG = {
    "enabled": True,
    "route_home_control": True,
    "strict_allowlist": True,
    "exposed_entities": "\n".join(
        (
            "weather.erzurum_hava_durumu",
            "light.masa_lambasi",
            "light.gece_lambasi",
            "light.oturma_odasi_lamba",
            "light.oturma_odasi_led",
            "sensor.oturma_odasi_nem",
            "sensor.oturma_odasi_sicaklik",
            "switch.kahve_makinesi",
            "humidifier.nemlendirici",
        )
    ),
    "aliases": "\n".join(
        (
            "weather.erzurum_hava_durumu: Erzurum havası, hava durumu",
            "light.masa_lambasi: masa lambası, çalışma lambası",
            "light.gece_lambasi: gece lambası, başucu lambası",
            "light.oturma_odasi_lamba: oturma odası lambası, oturma odası tavan lambası, salon tavan lambası",
            "light.oturma_odasi_led: oturma odası ledi, salon ledi",
            "sensor.oturma_odasi_nem: oturma odası nemi, salon nemi",
            "sensor.oturma_odasi_sicaklik: oturma odası derecesi, oturma odası sıcaklığı, salon sıcaklığı",
            "switch.kahve_makinesi: kahve makinesi",
            "humidifier.nemlendirici: nemlendirici, hava nemlendirici",
        )
    ),
}


HA_STATES = [
    {
        "entity_id": "weather.erzurum_hava_durumu",
        "state": "sunny",
        "friendly_name": "Erzurum Hava Durumu",
        "attributes": {"friendly_name": "Erzurum Hava Durumu", "temperature": 18},
    },
    {
        "entity_id": "light.masa_lambasi",
        "state": "off",
        "friendly_name": "Masa Lambası",
        "attributes": {"friendly_name": "Masa Lambası"},
    },
    {
        "entity_id": "light.gece_lambasi",
        "state": "on",
        "friendly_name": "Gece Lambası",
        "attributes": {"friendly_name": "Gece Lambası"},
    },
    {
        "entity_id": "light.oturma_odasi_lamba",
        "state": "on",
        "friendly_name": "Oturma Odası Lamba",
        "attributes": {"friendly_name": "Oturma Odası Lamba"},
    },
    {
        "entity_id": "light.oturma_odasi_led",
        "state": "on",
        "friendly_name": "Oturma Odası Led",
        "attributes": {"friendly_name": "Oturma Odası Led"},
    },
    {
        "entity_id": "sensor.oturma_odasi_nem",
        "state": "42",
        "friendly_name": "Oturma Odası Nem",
        "attributes": {"friendly_name": "Oturma Odası Nem", "unit_of_measurement": "%"},
    },
    {
        "entity_id": "sensor.oturma_odasi_sicaklik",
        "state": "22.4",
        "friendly_name": "Oturma Odası Sıcaklık",
        "attributes": {"friendly_name": "Oturma Odası Sıcaklık", "unit_of_measurement": "°C"},
    },
    {
        "entity_id": "switch.kahve_makinesi",
        "state": "off",
        "friendly_name": "Kahve Makinesi",
        "attributes": {"friendly_name": "Kahve Makinesi"},
    },
    {
        "entity_id": "humidifier.nemlendirici",
        "state": "off",
        "friendly_name": "Nemlendirici",
        "attributes": {"friendly_name": "Nemlendirici"},
    },
]


class FakeConfigStore:
    async def get(self, include_secrets: bool = True) -> dict[str, Any]:
        return {"ha_bridge": HA_CONFIG}


class FakeLogBus:
    def __init__(self) -> None:
        self.entries: list[tuple[Any, ...]] = []

    async def emit(self, *args: Any, **kwargs: Any) -> None:
        self.entries.append((*args, kwargs))


class FakeHomeAssistantBridge(HomeAssistantBridge):
    def __init__(self) -> None:
        super().__init__(FakeConfigStore(), FakeLogBus())
        self.service_calls: list[tuple[str, str, dict[str, Any]]] = []

    async def is_ready(self) -> bool:
        return True

    async def list_states(self, domain: str = "", limit: int = 64) -> list[dict[str, Any]]:
        rows = [
            dict(item)
            for item in HA_STATES
            if not domain or str(item["entity_id"]).startswith(f"{domain}.")
        ]
        return rows[:limit]

    async def get_state(self, entity_id: str) -> dict[str, Any] | None:
        return next((dict(item) for item in HA_STATES if item["entity_id"] == entity_id), None)

    async def get_weather_forecast_response(self, entity_id: str) -> dict[str, Any]:
        return {}

    async def call_service(self, domain: str, service: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        self.service_calls.append((domain, service, dict(data or {})))
        return {"ok": True}


class HaResponseComposerTests(unittest.TestCase):
    def test_shuffle_bag_avoids_repeating_control_templates(self) -> None:
        composer = HaResponseComposer(random.Random(7))
        responses = [composer.control("Masa lambası", "turn_on", "light") for _ in range(6)]
        self.assertEqual(6, len(set(responses)))
        next_response = composer.control("Masa lambası", "turn_on", "light")
        self.assertNotEqual(responses[-1], next_response)

    def test_control_response_contains_no_technical_home_assistant_syntax(self) -> None:
        composer = HaResponseComposer(random.Random(3))
        response = composer.control("Masa lambası", "set_color", "light", color_name="mavi")
        lowered = response.lower()
        self.assertNotIn("entity_id", lowered)
        self.assertNotIn("turn_on", lowered)
        self.assertNotIn("home assistant", lowered)

    def test_fake_home_assistant_tool_output_is_blocked(self) -> None:
        fake = (
            'Lambayı açıyorum. <alice_control_panel>'
            '{"action":"homeassistant.service_call","domain":"light","service":"turn_on"}'
        )
        safe = sanitize_assistant_output(fake)
        self.assertNotIn("alice_control_panel", safe)
        self.assertNotIn("service_call", safe)
        self.assertIn("güvenli", safe)


class HomeAssistantBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bridge = FakeHomeAssistantBridge()

    async def test_routes_punctuated_and_compact_turkish_commands(self) -> None:
        for text in ("Masa lambasını aç.", "masalambasını açar mısın?"):
            with self.subTest(text=text):
                self.assertTrue(await self.bridge.should_route_home_control(text))
                result = await self.bridge.handle_text_command(text)
                self.assertTrue(result["handled"])
                self.assertTrue(result["ok"])
                self.assertEqual("light.masa_lambasi", result["entity_id"])

    async def test_light_color_and_brightness_intents(self) -> None:
        cases = (
            ("Oturma odasını kırmızı yap.", "set_color", "kirmizi", None, None),
            ("Salon ışığını yüzde 30 yap.", "set_brightness", "", 30, None),
            ("Masa lambasını kıs.", "set_brightness", "", None, -25),
            ("Işığı biraz aç.", "set_brightness", "", None, 15),
            ("Işığı loş yap.", "set_brightness", "", 20, None),
            ("Ledleri mavi yap.", "set_color", "mavi", None, None),
            ("Sıcak beyaz yap.", "set_color", "sicak beyaz", None, None),
            ("Soğuk beyaz yap.", "set_color", "soguk beyaz", None, None),
            ("Renkleri normale al.", "set_color", "normal", None, None),
            ("Işıkları full aç.", "set_brightness", "", 100, None),
        )
        for text, action, color, brightness, step in cases:
            with self.subTest(text=text):
                intent = self.bridge.parse_intent(text)
                self.assertEqual(action, intent.action)
                self.assertEqual(color, intent.color_name)
                self.assertEqual(brightness, intent.brightness_pct)
                self.assertEqual(step, intent.brightness_step_pct)

    async def test_partial_alias_reserves_home_control_without_claiming_chat(self) -> None:
        self.assertTrue(await self.bridge.is_home_control_candidate("masa", partial=True))
        self.assertTrue(await self.bridge.is_home_control_candidate("Alice masa", partial=True))
        self.assertFalse(await self.bridge.should_route_home_control("masa"))
        self.assertFalse(await self.bridge.should_route_home_control("Bugün nasıl gidiyor?"))
        self.assertFalse(await self.bridge.should_route_home_control("Nemlendirici çok güzel."))

    async def test_ambiguous_lamp_requests_clarification(self) -> None:
        result = await self.bridge.handle_text_command("Lambayı kapat.")
        self.assertTrue(result["handled"])
        self.assertFalse(result["ok"])
        self.assertTrue(result["requires_clarification"])
        self.assertIn("Hang", result["speech"])
        self.assertEqual([], self.bridge.service_calls)

    async def test_room_bulk_command_targets_only_allowlisted_room_lights(self) -> None:
        result = await self.bridge.handle_text_command("Oturma odası ışıklarını kapat.")
        self.assertTrue(result["ok"])
        self.assertEqual(
            ["light.oturma_odasi_lamba", "light.oturma_odasi_led"],
            result["entity_ids"],
        )
        _, service, data = self.bridge.service_calls[-1]
        self.assertEqual("turn_off", service)
        self.assertEqual(result["entity_ids"], data["entity_id"])

    async def test_singular_room_lamp_does_not_control_other_room_lights(self) -> None:
        result = await self.bridge.handle_text_command("Oturma odası lambasını kapat.")
        self.assertTrue(result["ok"])
        self.assertEqual(["light.oturma_odasi_lamba"], result["entity_ids"])
        self.assertEqual("light.oturma_odasi_lamba", self.bridge.service_calls[-1][2]["entity_id"])

    async def test_plural_led_color_command_targets_only_allowlisted_leds(self) -> None:
        result = await self.bridge.handle_text_command("Ledleri mavi yap.")
        self.assertTrue(result["ok"])
        self.assertEqual(["light.oturma_odasi_led"], result["entity_ids"])
        domain, service, data = self.bridge.service_calls[-1]
        self.assertEqual("light", domain)
        self.assertEqual("turn_on", service)
        self.assertEqual([0, 90, 255], data["rgb_color"])

    async def test_room_color_without_singular_device_targets_room_lights(self) -> None:
        result = await self.bridge.handle_text_command("Oturma odasını kırmızı yap.")
        self.assertTrue(result["ok"])
        self.assertEqual(
            ["light.oturma_odasi_lamba", "light.oturma_odasi_led"],
            result["entity_ids"],
        )

    async def test_brightness_and_only_target_commands(self) -> None:
        brightness = await self.bridge.handle_text_command("Masa lambasını yüzde 30 yap.")
        self.assertTrue(brightness["ok"])
        self.assertEqual(30, self.bridge.service_calls[-1][2]["brightness_pct"])

        only_target = await self.bridge.handle_text_command("Sadece masa lambasını aç.")
        self.assertTrue(only_target["ok"])
        self.assertEqual("light.masa_lambasi", only_target["entity_id"])

    async def test_unqualified_all_command_asks_for_domain(self) -> None:
        result = await self.bridge.handle_text_command("Hepsini kapat.")
        self.assertTrue(result["handled"])
        self.assertFalse(result["ok"])
        self.assertTrue(result["requires_clarification"])
        self.assertIn("hangi grubu", result["speech"].lower())

    async def test_generic_switch_color_command_is_rejected(self) -> None:
        result = await self.bridge.handle_text_command("Prizi kırmızı yap.")
        self.assertTrue(result["handled"])
        self.assertFalse(result["ok"])
        self.assertIn("yalnızca ışıklarda", result["speech"])
        self.assertEqual([], self.bridge.service_calls)

    async def test_humidifier_has_its_own_domain(self) -> None:
        intent = self.bridge.parse_intent("Nemlendiriciyi aç.")
        self.assertEqual("humidifier", intent.domain_hint)
        result = await self.bridge.handle_text_command("Nemlendiriciyi aç.")
        self.assertTrue(result["ok"])
        self.assertEqual("humidifier.nemlendirici", result["entity_id"])
        self.assertEqual("humidifier", self.bridge.service_calls[-1][0])

    async def test_sensor_metric_words_disambiguate_temperature_and_humidity(self) -> None:
        temperature = await self.bridge.handle_text_command("Oturma odası kaç derece?")
        self.assertTrue(temperature["ok"])
        self.assertEqual("sensor.oturma_odasi_sicaklik", temperature["entity_id"])

        humidity = await self.bridge.handle_text_command("Oturma odasının nemi kaç?")
        self.assertTrue(humidity["ok"])
        self.assertEqual("sensor.oturma_odasi_nem", humidity["entity_id"])

    async def test_color_request_on_switch_returns_domain_mismatch(self) -> None:
        result = await self.bridge.handle_text_command("Kahve makinesini kırmızı yap.")
        self.assertTrue(result["handled"])
        self.assertFalse(result["ok"])
        self.assertIn("yalnızca ışıklarda", result["speech"])
        self.assertEqual([], self.bridge.service_calls)

    async def test_already_satisfied_command_skips_service_call(self) -> None:
        result = await self.bridge.handle_text_command("Gece lambasını aç.")
        self.assertTrue(result["ok"])
        self.assertTrue(result["already_satisfied"])
        self.assertIn("zaten açık", result["speech"])
        self.assertEqual([], self.bridge.service_calls)

    async def test_missing_domain_does_not_offer_unrelated_entities(self) -> None:
        result = await self.bridge.handle_text_command("Televizyonu aç.")
        self.assertTrue(result["handled"])
        self.assertFalse(result["ok"])
        self.assertNotIn("lamba", result["speech"].lower())


if __name__ == "__main__":
    unittest.main()
