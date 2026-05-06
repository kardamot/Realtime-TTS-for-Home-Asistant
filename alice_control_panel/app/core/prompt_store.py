from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import yaml

from .config_store import ConfigStore
from .paths import PROMPTS_DIR, ensure_data_dirs


DEFAULT_PROMPTS: dict[str, dict[str, str]] = {
    "alice": {
        "name": "Alice",
        "description": "Main assistant personality and home-control prompt.",
        "prompt": (
            "Sen Alice'sin. Turkce konusan, zeki, pratik ve ev otomasyonu ile "
            "robot kontrolunu sakin bir sekilde yoneten bir asistansin. Kisa, "
            "dogal ve uygulanabilir cevap ver. Gerektiginde soru sor."
        ),
    },
    "debug": {
        "name": "Debug",
        "description": "Verbose diagnostic prompt for pipeline testing.",
        "prompt": (
            "Debug modundasin. Kullanici istegini, secilen providerlari ve olasi "
            "sistem durumunu net sekilde acikla. Belirsizse varsayimlarini belirt."
        ),
    },
    "minimal": {
        "name": "Minimal",
        "description": "Short responses for fast command testing.",
        "prompt": "Cok kisa ve net Turkce cevap ver. Gereksiz aciklama yapma.",
    },
}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "prompt"


class PromptStore:
    def __init__(self, config_store: ConfigStore, prompts_dir: Path = PROMPTS_DIR) -> None:
        self._config_store = config_store
        self._prompts_dir = prompts_dir

    async def ensure_defaults(self) -> None:
        ensure_data_dirs()
        self._prompts_dir.mkdir(parents=True, exist_ok=True)
        for slug, data in DEFAULT_PROMPTS.items():
            path = self._path(slug)
            if not path.exists():
                self._write(path, {**data, "slug": slug, "updated_at": time.time()})

    async def list_profiles(self) -> dict[str, Any]:
        config = await self._config_store.get(include_secrets=False)
        active = str(config.get("prompts", {}).get("active_profile") or "alice")
        profiles = []
        for path in sorted(self._prompts_dir.glob("*.yaml")):
            doc = self._read(path)
            slug = slugify(str(doc.get("slug") or path.stem))
            profiles.append(
                {
                    "slug": slug,
                    "name": str(doc.get("name") or slug.title()),
                    "description": str(doc.get("description") or ""),
                    "active": slug == active,
                    "updated_at": doc.get("updated_at"),
                }
            )
        return {"active_profile": active, "profiles": profiles}

    async def get_profile(self, slug: str) -> dict[str, Any]:
        slug = slugify(slug)
        path = self._path(slug)
        if not path.exists():
            raise FileNotFoundError(slug)
        doc = self._read(path)
        doc["slug"] = slug
        return doc

    async def save_profile(self, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
        slug = slugify(slug)
        doc = {
            "slug": slug,
            "name": str(payload.get("name") or slug.title()),
            "description": str(payload.get("description") or ""),
            "prompt": str(payload.get("prompt") or ""),
            "updated_at": time.time(),
        }
        self._write(self._path(slug), doc)
        return doc

    async def create_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        slug = slugify(str(payload.get("slug") or payload.get("name") or "prompt"))
        original = slug
        index = 2
        while self._path(slug).exists():
            slug = f"{original}-{index}"
            index += 1
        return await self.save_profile(slug, payload)

    async def copy_profile(self, source: str, new_name: str) -> dict[str, Any]:
        doc = await self.get_profile(source)
        doc["name"] = new_name
        doc["slug"] = slugify(new_name)
        return await self.create_profile(doc)

    async def delete_profile(self, slug: str) -> None:
        slug = slugify(slug)
        if slug in DEFAULT_PROMPTS:
            raise ValueError("Default prompt profiles cannot be deleted")
        path = self._path(slug)
        if path.exists():
            path.unlink()

    async def activate(self, slug: str) -> None:
        slug = slugify(slug)
        if not self._path(slug).exists():
            raise FileNotFoundError(slug)
        await self._config_store.set_active_prompt(slug)

    async def active_prompt_text(self) -> str:
        config = await self._config_store.get(include_secrets=True)
        slug = str(config.get("prompts", {}).get("active_profile") or "alice")
        try:
            doc = await self.get_profile(slug)
        except FileNotFoundError:
            doc = await self.get_profile("alice")
        return str(doc.get("prompt") or "")

    def _path(self, slug: str) -> Path:
        return self._prompts_dir / f"{slugify(slug)}.yaml"

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        return doc if isinstance(doc, dict) else {}

    @staticmethod
    def _write(path: Path, doc: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False)

