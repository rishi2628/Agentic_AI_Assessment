"""
Session Memory — in-process conversation buffer for the Policy & Incident Copilot.

Design goals (matching the problem statement)
----------------------------------------------
* Store only *safe* preferences and conversation context.
* NEVER persist:
    - Passwords, credentials, or secrets
    - Full SSNs or government IDs
    - Raw authentication tokens
* Automatically redact the above from any message before storing.
* Keep a rolling window (configurable) to respect token budgets.
* Provide a clean interface for agents to read/write context.

This is an in-memory implementation.  For production you would swap the
backing store to Redis or a database, but the interface stays the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Sensitive-content patterns ────────────────────────────────────────────────

# Full SSN  e.g.  123-45-6789  or  123456789
_SSN_RE = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
# Passwords in context: "password: abc123", "pwd=abc123"
_PASSWORD_RE = re.compile(
    r"(?i)\b(?:password|passwd|pwd|secret|token|api[-_]?key)\s*[=:]\s*\S+",
)
# Bearer / JWT tokens
_TOKEN_RE = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)

_REDACT_PATTERNS = [
    (_SSN_RE, "[SSN REDACTED]"),
    (_PASSWORD_RE, "[CREDENTIAL REDACTED]"),
    (_TOKEN_RE, "[TOKEN REDACTED]"),
]


def _sanitize(text: str) -> str:
    """Strip sensitive patterns from *text* before storing in memory."""
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str   # "user" | "assistant" | "system"
    content: str


@dataclass
class SessionMemory:
    """
    Rolling conversation buffer with automatic PII / credential sanitisation.

    Parameters
    ----------
    max_turns:
        Maximum number of *user + assistant* turn pairs to retain.
        Older pairs are dropped (FIFO) when the limit is exceeded.
        This prevents unbounded token growth in long sessions.
    """

    max_turns: int = 20
    _messages: List[Message] = field(default_factory=list, init=False, repr=False)

    # ── write ─────────────────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        """Record a user turn (sanitised before storage)."""
        self._messages.append(Message(role="user", content=_sanitize(content)))
        self._trim()

    def add_assistant(self, content: str) -> None:
        """Record an assistant turn (sanitised before storage)."""
        self._messages.append(Message(role="assistant", content=_sanitize(content)))
        self._trim()

    def add_system(self, content: str) -> None:
        """
        Record a system message (e.g. user preference like preferred language).
        Not counted against the rolling window.
        """
        # System messages are prepended and never trimmed
        self._messages.insert(0, Message(role="system", content=_sanitize(content)))

    # ── read ──────────────────────────────────────────────────────────────────

    @property
    def messages(self) -> List[Message]:
        """Return all messages (immutable view)."""
        return list(self._messages)

    def to_langchain_messages(self) -> List:
        """
        Convert to LangChain message objects for direct use in chains.
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        mapping = {
            "user": HumanMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
        }
        return [mapping[m.role](content=m.content) for m in self._messages]

    def last_n(self, n: int = 5) -> List[Message]:
        """Return the *n* most recent messages."""
        return self._messages[-n:]

    def clear(self) -> None:
        """Reset the conversation (e.g. when starting a new topic)."""
        # Keep system messages (preferences)
        self._messages = [m for m in self._messages if m.role == "system"]

    # ── internals ─────────────────────────────────────────────────────────────

    def _trim(self) -> None:
        """Enforce the rolling-window limit (system messages excluded)."""
        non_system = [m for m in self._messages if m.role != "system"]
        system = [m for m in self._messages if m.role == "system"]
        # Each "turn" = 1 user + 1 assistant message → max_turns * 2 messages
        if len(non_system) > self.max_turns * 2:
            non_system = non_system[-(self.max_turns * 2):]
        self._messages = system + non_system

    def __len__(self) -> int:
        return len(self._messages)
