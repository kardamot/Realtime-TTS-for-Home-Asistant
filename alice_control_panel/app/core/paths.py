from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ADDON_ROOT = PACKAGE_ROOT.parent

DATA_DIR = Path(os.getenv("ALICE_DATA_DIR", "/data"))
OPTIONS_PATH = Path(os.getenv("ALICE_OPTIONS_PATH", str(DATA_DIR / "options.json")))
CONFIG_PATH = Path(os.getenv("ALICE_CONFIG_PATH", str(DATA_DIR / "alice_config.json")))
PROMPTS_DIR = Path(os.getenv("ALICE_PROMPTS_DIR", str(DATA_DIR / "prompts")))
STATIC_DIR = Path(os.getenv("ALICE_STATIC_DIR", str(ADDON_ROOT / "static")))
FRONTEND_DIST_DIR = ADDON_ROOT / "frontend" / "dist"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

