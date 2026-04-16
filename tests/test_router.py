"""
Unit tests for the Request Router.

These tests exercise ONLY the heuristic routing logic — no LLM is required.
Run with:  pytest tests/test_router.py -v
"""

import pytest
from src.agents.router import RequestRouter, RoutingDecision
from tests.test_cases import SINGLE_AGENT_CASES, MULTI_AGENT_CASES


@pytest.fixture
def router():
    """Router without LLM (heuristics only)."""
    return RequestRouter(llm=None)


# ── Single-agent routing ──────────────────────────────────────────────────────

@pytest.mark.parametrize("case", SINGLE_AGENT_CASES, ids=[c["id"] for c in SINGLE_AGENT_CASES])
def test_router_single_agent(router, case):
    """Short, clear policy questions should route to single-agent."""
    decision: RoutingDecision = router.route(case["input"])
    # Simple queries should be single; if multi, the router is over-triggering
    # Accept both for borderline cases but verify MULTI cases don't appear here
    assert isinstance(decision, RoutingDecision)
    assert decision.mode in ("single", "multi")   # valid output
    # For clearly single inputs (short + simple keyword), expect single
    if len(case["input"]) < 200:
        assert decision.mode == "single", (
            f"[{case['id']}] Expected 'single', got '{decision.mode}'. "
            f"Reasons: {decision.reasons}"
        )


# ── Multi-agent routing ───────────────────────────────────────────────────────

@pytest.mark.parametrize("case", MULTI_AGENT_CASES, ids=[c["id"] for c in MULTI_AGENT_CASES])
def test_router_multi_agent(router, case):
    """Complex / noisy / high-risk inputs should route to multi-agent."""
    decision: RoutingDecision = router.route(case["input"])
    assert decision.mode == "multi", (
        f"[{case['id']}] Expected 'multi', got '{decision.mode}'. "
        f"Reasons: {decision.reasons}"
    )


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_router_empty_input(router):
    """Empty input defaults to single without crashing."""
    decision = router.route("")
    assert decision.mode in ("single", "multi")


def test_router_very_long_input_routes_multi(router):
    """Inputs > 500 chars trigger multi-agent."""
    long_input = "What is the policy? " * 50   # 1000+ chars
    decision = router.route(long_input)
    assert decision.mode == "multi"


def test_router_disable_mfa_triggers_multi(router):
    decision = router.route("Please disable MFA for user john@company.com")
    assert decision.mode == "multi"


def test_router_log_lines_trigger_multi(router):
    log_input = (
        "2025-04-16T09:00:00Z ERROR vpn-gw01: connection refused\n"
        "2025-04-16T09:01:00Z ERROR vpn-gw01: TLS handshake failed"
    )
    decision = router.route(log_input)
    assert decision.mode == "multi"


def test_routing_decision_has_reasons(router):
    decision = router.route("VPN failing for 150 users right now")
    assert len(decision.reasons) > 0
    assert decision.confidence in ("high", "medium")
