from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


VALID_LEVELS = {"DEBUG", "INFO", "WARN", "ERROR"}
CRITICAL_LEVELS = {"WARN", "ERROR"}
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LogEntry:
    id: str
    ts: float
    level: str
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LogBus:
    def __init__(
        self,
        maxlen: int = 1000,
        critical_dir: Path | None = None,
        critical_retention_days: int = 31,
        critical_max_total_bytes: int = 32 * 1024 * 1024,
        critical_max_file_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        self._entries: deque[LogEntry] = deque(maxlen=maxlen)
        self._subscribers: set[asyncio.Queue[LogEntry]] = set()
        self._lock = asyncio.Lock()
        self._critical_dir = Path(critical_dir) if critical_dir else None
        self._critical_retention_days = max(1, int(critical_retention_days))
        self._critical_max_total_bytes = max(1024, int(critical_max_total_bytes))
        self._critical_max_file_bytes = max(1024, int(critical_max_file_bytes))
        self._last_archive_prune_at = 0.0
        self._prepare_critical_archive()

    async def emit(
        self,
        level: str,
        category: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> LogEntry:
        level = level.upper()
        if level not in VALID_LEVELS:
            level = "INFO"
        entry = LogEntry(
            id=uuid4().hex,
            ts=time.time(),
            level=level,
            category=category.upper(),
            message=message,
            details=details or {},
        )
        async with self._lock:
            self._entries.append(entry)
            if entry.level in CRITICAL_LEVELS:
                self._append_critical_entry(entry)
            subscribers = list(self._subscribers)
        for queue in subscribers:
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                try:
                    _ = queue.get_nowait()
                    queue.put_nowait(entry)
                except asyncio.QueueEmpty:
                    pass
        return entry

    def emit_nowait(
        self,
        level: str,
        category: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.emit(level, category, message, details))

    async def list(
        self,
        level: str | None = None,
        category: str | None = None,
        search: str | None = None,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        levels = {item.strip().upper() for item in level.split(",")} if level else set()
        categories = {item.strip().upper() for item in category.split(",")} if category else set()
        query = (search or "").strip().lower()
        async with self._lock:
            entries = list(self._entries)
        filtered: list[LogEntry] = []
        for entry in entries:
            if levels and entry.level not in levels:
                continue
            if categories and entry.category not in categories:
                continue
            if query and query not in entry.message.lower() and query not in json.dumps(entry.details).lower():
                continue
            filtered.append(entry)
        return [entry.to_dict() for entry in filtered[-max(1, min(limit, 1000)):]]

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._clear_critical_archive()

    async def download_text(self) -> str:
        async with self._lock:
            entries = self._merge_entries(self._read_critical_entries(), list(self._entries))
        lines = []
        for entry in entries:
            details = f" {json.dumps(entry.details, ensure_ascii=False)}" if entry.details else ""
            lines.append(f"{entry.ts:.3f} [{entry.level}] {entry.category}: {entry.message}{details}")
        return "\n".join(lines) + ("\n" if lines else "")

    async def subscribe(self) -> asyncio.Queue[LogEntry]:
        queue: asyncio.Queue[LogEntry] = asyncio.Queue(maxsize=250)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[LogEntry]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    def archive_status(self) -> dict[str, Any]:
        paths = self._critical_paths()
        total_bytes = sum(self._safe_file_size(path) for path in paths)
        return {
            "enabled": self._critical_dir is not None,
            "retention_days": self._critical_retention_days,
            "max_total_bytes": self._critical_max_total_bytes,
            "max_file_bytes": self._critical_max_file_bytes,
            "file_count": len(paths),
            "total_bytes": total_bytes,
        }

    def _prepare_critical_archive(self) -> None:
        if self._critical_dir is None:
            return
        try:
            self._critical_dir.mkdir(parents=True, exist_ok=True)
            self._prune_critical_archive(force=True)
            for entry in self._read_critical_entries():
                self._entries.append(entry)
        except OSError as exc:
            LOGGER.warning("Critical log archive initialization failed: %s", exc)

    def _append_critical_entry(self, entry: LogEntry) -> None:
        if self._critical_dir is None:
            return
        try:
            encoded = self._encode_archive_entry(entry)
            path = self._archive_path_for(entry.ts, len(encoded))
            with path.open("ab") as handle:
                handle.write(encoded)
            self._prune_critical_archive()
        except (OSError, TypeError, ValueError) as exc:
            LOGGER.warning("Critical log archive write failed: %s", exc)

    def _encode_archive_entry(self, entry: LogEntry) -> bytes:
        encoded = (json.dumps(entry.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(encoded) <= self._critical_max_file_bytes:
            return encoded
        compact = entry.to_dict()
        compact["message"] = str(compact.get("message") or "")[:4000]
        compact["details"] = {"archive_note": "oversized details omitted"}
        return (json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")

    def _archive_path_for(self, timestamp: float, incoming_bytes: int) -> Path:
        assert self._critical_dir is not None
        day = time.strftime("%Y-%m-%d", time.gmtime(timestamp))
        part = 0
        while True:
            path = self._critical_dir / f"critical-{day}-{part:03d}.jsonl"
            if not path.exists() or self._safe_file_size(path) + incoming_bytes <= self._critical_max_file_bytes:
                return path
            part += 1

    def _prune_critical_archive(self, force: bool = False) -> None:
        if self._critical_dir is None:
            return
        now = time.time()
        prune_by_age = force or now - self._last_archive_prune_at >= 6 * 60 * 60
        if prune_by_age:
            self._last_archive_prune_at = now
            cutoff = now - self._critical_retention_days * 24 * 60 * 60
            paths = self._critical_paths()
            for path in paths:
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                except OSError:
                    continue

        paths = sorted(self._critical_paths(), key=self._safe_file_mtime)
        total_bytes = sum(self._safe_file_size(path) for path in paths)
        for path in paths:
            if total_bytes <= self._critical_max_total_bytes:
                break
            size = self._safe_file_size(path)
            try:
                path.unlink(missing_ok=True)
                total_bytes -= size
            except OSError:
                continue

    def _clear_critical_archive(self) -> None:
        for path in self._critical_paths():
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("Critical log archive clear failed for %s: %s", path, exc)

    def _read_critical_entries(self) -> list[LogEntry]:
        entries: list[LogEntry] = []
        for path in self._critical_paths():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        entry = self._decode_archive_entry(raw)
                        if entry is not None:
                            entries.append(entry)
            except (OSError, UnicodeError) as exc:
                LOGGER.warning("Critical log archive read failed for %s: %s", path, exc)
        entries.sort(key=lambda item: item.ts)
        return entries

    @staticmethod
    def _decode_archive_entry(raw: str) -> LogEntry | None:
        try:
            doc = json.loads(raw)
            level = str(doc.get("level") or "").upper()
            if level not in CRITICAL_LEVELS:
                return None
            details = doc.get("details") if isinstance(doc.get("details"), dict) else {}
            return LogEntry(
                id=str(doc.get("id") or uuid4().hex),
                ts=float(doc.get("ts") or 0),
                level=level,
                category=str(doc.get("category") or "SYSTEM").upper(),
                message=str(doc.get("message") or ""),
                details=details,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _critical_paths(self) -> list[Path]:
        if self._critical_dir is None or not self._critical_dir.exists():
            return []
        return sorted(self._critical_dir.glob("critical-*.jsonl"))

    @staticmethod
    def _merge_entries(*groups: list[LogEntry]) -> list[LogEntry]:
        merged: dict[str, LogEntry] = {}
        for group in groups:
            for entry in group:
                merged[entry.id] = entry
        return sorted(merged.values(), key=lambda item: item.ts)

    @staticmethod
    def _safe_file_size(path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    @staticmethod
    def _safe_file_mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0
