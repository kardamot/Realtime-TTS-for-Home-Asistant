from __future__ import annotations

import random
from typing import Any, Sequence


class _ShuffleBag:
    def __init__(self, rng: Any | None = None) -> None:
        self._rng = rng or random.SystemRandom()
        self._bags: dict[str, list[int]] = {}
        self._last: dict[str, int] = {}

    def choose(self, key: str, values: Sequence[str]) -> str:
        if not values:
            return ""
        if len(values) == 1:
            return values[0]

        bag = self._bags.get(key)
        if not bag or any(index >= len(values) for index in bag):
            bag = list(range(len(values)))
            self._rng.shuffle(bag)
            last = self._last.get(key)
            if last is not None and bag[-1] == last:
                bag[0], bag[-1] = bag[-1], bag[0]
            self._bags[key] = bag

        index = bag.pop()
        self._last[key] = index
        return values[index]


class HaResponseComposer:
    """Builds short factual Turkish HA replies without an LLM round trip."""

    def __init__(self, rng: Any | None = None) -> None:
        self._variants = _ShuffleBag(rng)

    def control(
        self,
        target: str,
        action: str,
        domain: str,
        *,
        count: int = 1,
        color_name: str = "",
        brightness_pct: int | None = None,
        brightness_step_pct: int | None = None,
        temperature: float | None = None,
        hvac_mode: str = "",
    ) -> str:
        target = target.strip() or ("Cihazlar" if count > 1 else "Cihaz")
        key = f"{domain}:{action}:{'group' if count > 1 else 'single'}"

        if action == "turn_on":
            if domain == "light":
                inline_target = self._inline_name(target)
                light_variants = (
                    (
                        f"{target} açık.",
                        f"{target} şimdi açık.",
                        f"Tamam, {inline_target} açık.",
                        f"Oldu, {inline_target} açıldı.",
                        f"Işıklar hazır; {inline_target} açık.",
                        f"{target} açıldı, karanlık biraz geri çekildi.",
                    )
                    if count > 1
                    else (
                        f"{target} açık.",
                        f"{target} şimdi açık.",
                        f"Tamam, {inline_target} açık.",
                        f"Oldu, {inline_target} açıldı.",
                        f"{target} hazır; ışığı yandı.",
                        f"{target} açıldı, karanlık biraz geri çekildi.",
                    )
                )
                return self._pick(
                    key,
                    light_variants,
                )
            if domain == "switch":
                return self._pick(
                    key,
                    (
                        f"{target} açık.",
                        f"{target} devrede.",
                        f"Tamam, {target} çalışıyor.",
                        f"Oldu, {target} açıldı.",
                        f"{target} şimdi etkin.",
                    ),
                )
            if domain == "humidifier":
                return self._pick(
                    key,
                    (
                        f"{target} çalışıyor.",
                        f"{target} açıldı.",
                        f"Tamam, {target} devrede.",
                        f"Oldu, {target} çalışmaya başladı.",
                    ),
                )
            return self._pick(
                key,
                (
                    f"{target} açıldı.",
                    f"Tamam, {target} çalışıyor.",
                    f"Oldu, {target} devrede.",
                    f"{target} şimdi etkin.",
                ),
            )

        if action == "turn_off":
            return self._pick(
                key,
                (
                    f"{target} kapalı.",
                    f"{target} artık kapalı.",
                    f"Tamam, {target} kapatıldı.",
                    f"Oldu, {target} kapalı.",
                    f"{target} devreden çıktı.",
                    f"{target} kapandı; biraz sakinlik iyi gelir.",
                ),
            )

        if action == "toggle":
            return self._pick(
                key,
                (
                    f"{target} durumu değiştirildi.",
                    f"{target} için durumu çevirdim.",
                    f"Tamam, {target} diğer duruma geçti.",
                ),
            )

        if action == "set_color":
            color = color_name.strip() or "istenen renk"
            return self._pick(
                key,
                (
                    f"{target} artık {color}.",
                    f"{target} rengini {color} yaptım.",
                    f"{target} için {color} tonu hazır.",
                    f"Tamam, {target} {color} oldu.",
                    f"{color.capitalize()} tamam; {target} hazır.",
                ),
            )

        if action == "set_brightness":
            if brightness_pct is not None:
                return self._pick(
                    f"{key}:absolute",
                    (
                        f"{target} yüzde {brightness_pct} parlaklıkta.",
                        f"{target} parlaklığını yüzde {brightness_pct} yaptım.",
                        f"Tamam, {target} yüzde {brightness_pct} seviyesinde.",
                        f"{target} için parlaklık yüzde {brightness_pct} oldu.",
                    ),
                )
            if brightness_step_pct is not None and brightness_step_pct < 0:
                return self._pick(
                    f"{key}:dimmer",
                    (
                        f"{target} biraz daha loş.",
                        f"{target} ışığını biraz azalttım.",
                        f"Tamam, {target} biraz kısıldı.",
                        f"{target} daha yumuşak bir seviyede.",
                    ),
                )
            return self._pick(
                f"{key}:brighter",
                (
                    f"{target} biraz daha parlak.",
                    f"{target} ışığını biraz yükselttim.",
                    f"Tamam, {target} biraz açıldı.",
                    f"{target} için parlaklığı artırdım.",
                ),
            )

        if action == "set_temperature" and temperature is not None:
            value = self._number(temperature)
            return self._pick(
                key,
                (
                    f"{target} {value} dereceye ayarlandı.",
                    f"{target} için sıcaklık {value} derece.",
                    f"Tamam, {target} {value} dereceye geçiyor.",
                    f"{value} derece tamam; {target} ayarlandı.",
                ),
            )

        if action == "set_hvac":
            mode = {
                "heat": "ısıtma",
                "cool": "soğutma",
                "auto": "otomatik",
            }.get(hvac_mode, "istenen")
            return self._pick(
                key,
                (
                    f"{target} {mode} modunda.",
                    f"{target} için {mode} modunu seçtim.",
                    f"Tamam, {target} {mode} moduna geçti.",
                ),
            )

        if action == "set_media_volume" and brightness_pct is not None:
            return self._pick(
                key,
                (
                    f"{target} sesi yüzde {brightness_pct}.",
                    f"{target} sesini yüzde {brightness_pct} yaptım.",
                    f"Tamam, {target} yüzde {brightness_pct} ses seviyesinde.",
                ),
            )

        return self._pick(
            key,
            (
                f"{target} için komut uygulandı.",
                f"Tamam, {target} hazır.",
                f"Oldu, {target} ayarlandı.",
            ),
        )

    def state(self, target: str, value: str, domain: str, unit: str = "") -> str:
        target = target.strip() or "Cihaz"
        if domain in {"light", "switch", "fan", "input_boolean", "humidifier"} and value in {"on", "off"}:
            state_text = "açık" if value == "on" else "kapalı"
            return self._pick(
                f"state:{domain}:{value}",
                (
                    f"{target} şu anda {state_text}.",
                    f"{target} {state_text} görünüyor.",
                    f"Kontrol ettim, {target} {state_text}.",
                ),
            )
        suffix = f" {unit}" if unit else ""
        return f"{target}: {value}{suffix}."

    def already(self, target: str, value: str) -> str:
        state_text = "açık" if value == "on" else "kapalı"
        return self._pick(
            f"already:{value}",
            (
                f"{target} zaten {state_text}.",
                f"Kontrol ettim, {target} zaten {state_text}.",
                f"{target} için değişiklik yok; zaten {state_text}.",
            ),
        )

    def clarification(self, names: Sequence[str]) -> str:
        clean = list(dict.fromkeys(name.strip() for name in names if name.strip()))[:3]
        if not clean:
            return self._pick(
                "clarification:none",
                (
                    "Hangi cihazı kastettiğini biraz daha net söyler misin?",
                    "Cihaz adını biraz daha açık söyler misin?",
                    "Hedefi netleştirelim; hangi cihazı istiyorsun?",
                ),
            )
        names_text = ", ".join(clean)
        if len(clean) == 1:
            particle = self._question_particle(names_text)
            return self._pick(
                "clarification:single",
                (
                    f"{names_text} {particle} demek istiyorsun?",
                    f"Hedefin {names_text} {particle}?",
                ),
            )
        return self._pick(
            "clarification:multiple",
            (
                f"Birden fazla seçenek var: {names_text}. Hangisini istiyorsun?",
                f"Şunu netleştirelim: {names_text}. Hangisi?",
                f"{names_text} arasında kaldım. Hangisini kastediyorsun?",
            ),
        )

    def no_match(self, *, has_area: bool, names: Sequence[str] = ()) -> str:
        clean = list(dict.fromkeys(name.strip() for name in names if name.strip()))[:3]
        if clean:
            names_text = ", ".join(clean)
            return self._pick(
                f"no_match:{'area' if has_area else 'target'}:alternatives",
                (
                    f"Bu adla uygun cihaz bulamadım. Yakın seçenekler: {names_text}.",
                    f"Hedef net eşleşmedi. Şunlardan biri olabilir: {names_text}.",
                ),
            )
        if has_area:
            return self._pick(
                "no_match:area",
                (
                    "Bu oda için kontrol edebileceğim uygun bir cihaz bulamadım.",
                    "Bu odada komuta uyan izinli bir cihaz görünmüyor.",
                ),
            )
        return self._pick(
            "no_match:target",
            (
                "Bu adla kontrol edebileceğim bir cihaz bulamadım.",
                "Söylediğin cihazı izinli cihazlar arasında bulamadım.",
            ),
        )

    def domain_mismatch(self, target: str, action: str) -> str:
        if action in {"set_color", "set_brightness"}:
            return f"Bunu {target} üzerinde yapamam; renk ve parlaklık yalnızca ışıklarda kullanılabilir."
        if action in {"set_temperature", "set_hvac"}:
            return f"Bunu {target} üzerinde yapamam; hedef bir klima veya termostat değil."
        if action == "set_media_volume":
            return f"Bunu {target} üzerinde yapamam; hedef bir medya oynatıcı değil."
        return f"Bu komut {target} için uygun değil."

    def route_error(self, kind: str = "connection") -> str:
        if kind == "permission":
            return "Bu cihaz için kontrol iznim yok."
        if kind == "transcript":
            return self._pick(
                "route_error:transcript",
                (
                    "Komutun sonunu net duyamadım. Cihazı ve işlemi tekrar söyler misin?",
                    "Ev komutu gibi duydum ama cümle tamamlanmadı. Bir kez daha söyler misin?",
                ),
            )
        return self._pick(
            "route_error:connection",
            (
                "Ev kontrolüne şu anda ulaşamadım. Birazdan yeniden deneyelim.",
                "Cihaza ulaşırken kısa bir bağlantı sorunu oldu. Yeniden deneyelim.",
            ),
        )

    def _pick(self, key: str, values: Sequence[str]) -> str:
        return self._variants.choose(key, values)

    @staticmethod
    def _number(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return str(value).replace(".", ",")

    @staticmethod
    def _question_particle(value: str) -> str:
        for char in reversed(value.lower()):
            if char in "aı":
                return "mı"
            if char in "ei":
                return "mi"
            if char in "ou":
                return "mu"
            if char in "öü":
                return "mü"
        return "mı"

    @staticmethod
    def _inline_name(value: str) -> str:
        if not value:
            return value
        return value[0].lower() + value[1:]
