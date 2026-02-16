from __future__ import annotations

from collections import defaultdict


class SessionMemory:
    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, str]]] = defaultdict(list)

    def add(self, session_id: str, role: str, content: str) -> None:
        self._messages[session_id].append({"role": role, "content": content})

    def get(self, session_id: str) -> list[dict[str, str]]:
        return self._messages.get(session_id, [])

    def clear(self, session_id: str) -> None:
        self._messages.pop(session_id, None)


memory = SessionMemory()

