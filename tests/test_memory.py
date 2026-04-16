"""
Unit tests for SessionMemory — no LLM required.

Run with:  pytest tests/test_memory.py -v
"""

import pytest
from src.memory.session_memory import SessionMemory
from tests.test_cases import MEMORY_CASES


@pytest.fixture
def mem():
    return SessionMemory(max_turns=5)


# ── PII redaction in memory ───────────────────────────────────────────────────

@pytest.mark.parametrize("case", MEMORY_CASES, ids=[c["id"] for c in MEMORY_CASES])
def test_pii_not_stored_in_memory(mem, case):
    """
    Sensitive patterns (SSN, passwords, tokens) must be redacted before storage.
    After adding the message, retrieve it and assert the raw PII is gone.
    """
    mem.add_user(case["input"])
    stored = mem.messages[-1].content

    # SSN pattern
    import re
    ssn_re = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    assert not ssn_re.search(stored), f"SSN found in stored memory: {stored!r}"

    # Credential (password=value or password:value — not "password policy")
    pwd_re = re.compile(r"(?i)\bpassword\s*[=:]\s*\S{4,}")
    assert not pwd_re.search(stored), f"Password found in stored memory: {stored!r}"

    # Bearer token pattern
    token_re = re.compile(r"Bearer\s+[A-Za-z0-9]{8,}", re.I)
    assert not token_re.search(stored), f"Token found in stored memory: {stored!r}"


# ── Rolling window ────────────────────────────────────────────────────────────

def test_rolling_window_respected(mem):
    """Memory should not grow beyond max_turns * 2 non-system messages."""
    for i in range(20):
        mem.add_user(f"Question {i}")
        mem.add_assistant(f"Answer {i}")

    non_system = [m for m in mem.messages if m.role != "system"]
    assert len(non_system) <= mem.max_turns * 2


def test_system_messages_not_trimmed(mem):
    """System messages (preferences) survive the rolling trim."""
    mem.add_system("User prefers responses in bullet points.")
    for i in range(20):
        mem.add_user(f"q{i}")
        mem.add_assistant(f"a{i}")

    system_msgs = [m for m in mem.messages if m.role == "system"]
    assert len(system_msgs) == 1


# ── Clear resets only non-system messages ────────────────────────────────────

def test_clear_keeps_system_messages(mem):
    mem.add_system("Preferred language: English")
    mem.add_user("What is the password policy?")
    mem.add_assistant("Passwords expire every 90 days.")
    mem.clear()

    assert len(mem) == 1
    assert mem.messages[0].role == "system"


# ── LangChain message conversion ─────────────────────────────────────────────

def test_to_langchain_messages(mem):
    from langchain_core.messages import HumanMessage, AIMessage

    mem.add_user("What is the VPN endpoint?")
    mem.add_assistant("The VPN endpoint is vpn.company.com.")

    lc_msgs = mem.to_langchain_messages()
    assert any(isinstance(m, HumanMessage) for m in lc_msgs)
    assert any(isinstance(m, AIMessage) for m in lc_msgs)


# ── last_n helper ─────────────────────────────────────────────────────────────

def test_last_n_returns_correct_count(mem):
    for i in range(10):
        mem.add_user(f"q{i}")
    last = mem.last_n(3)
    assert len(last) == 3
    assert last[-1].content == "q9"
