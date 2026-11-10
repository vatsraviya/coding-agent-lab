"""A predetermined answer-card client used for repeatable repair-loop tests."""

from __future__ import annotations


class ScriptedClient:
    """Return pre-written replies in order. Does not call a model or network."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the next prepared reply, or raise if the card stack is empty."""

        del messages  # The script ignores conversation content by design.
        if self._index >= len(self._replies):
            raise RuntimeError("ScriptedClient has no remaining replies.")
        reply = self._replies[self._index]
        self._index += 1
        return reply
