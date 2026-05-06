from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


VALID_LEVELS = {"DEBUG", "INFO", "WARN", "ERROR"}


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
    def __init__(self, maxlen: int = 1000) -> None:
        self._entries: deque[LogEntry] = deque(maxlen=maxlen)
        self._subscribers: set[asyncio.Queue[LogEntry]] = set()
        self._lock = asyncio.Lock()

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

    async def download_text(self) -> str:
        async with self._lock:
            entries = list(self._entries)
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

