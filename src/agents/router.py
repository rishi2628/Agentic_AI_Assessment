"""
Request Router — decides whether a user request should be handled by
the Single-Agent or the Multi-Agent pipeline.

Routing logic
-------------
The router uses a two-pass approach:

Pass 1 — Deterministic heuristics (fast, no LLM call needed)
  Multi-agent triggers:
  * Input length > 500 characters  (implies long log / conflicting info)
  * User count ≥ 10 mentioned      (widespread incident → P1/P2)
  * Safety/override keywords detected (disable MFA, ignore policy, …)
  * Explicit log pasted (lines starting with timestamps or log-level prefixes)

  Single-agent triggers:
  * Short, clear question with a single topic
  * Keyword: "what is", "summarize", "how do I", "who do I contact"

Pass 2 — LLM confirmation (optional, triggered when heuristics are ambiguous)
  A cheap single-turn LLM call with a constrained prompt that returns
  only "SINGLE" or "MULTI".  This pass is SKIPPED if heuristics are confident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.logger import CopilotLogger

_logger = CopilotLogger("router")

# ── Heuristic thresholds ─────────────────────────────────────────────────────

_MULTI_LENGTH_THRESHOLD = 500       # characters
_MULTI_USER_THRESHOLD = 10          # users mentioned

_MULTI_KEYWORDS = [
    r"disable\s+mfa",
    r"turn\s+off\s+mfa",
    r"bypass\s+mfa",
    r"ignore\s+(the\s+)?(?:\w+\s+)*policy",
    r"override\s+(the\s+)?(?:policy|security)",
    r"\d{2,}[\s,]*users?\s+(?:affected|failing|unable|cannot)",
    r"\d{1,3}\s+users?\s+(?:are\s+)?(?:reporting|experiencing|affected|failing|unable|cannot)",
    r"vpn\s+(?:is\s+)?(?:down|failing|broken|not\s+working)\s+for",
    r"outage",
    r"multiple\s+(?:users?|employees?|machines?)",
    r"conflicting|inconsistent|two\s+different\s+(?:staff|it|sources?|answers?)",
    r"either.{1,30}or.{1,30}days",   # conflicting durations
    r"error\s+log",
    r"exception\s+trace",
    r"stack\s+trace",
]

_LOG_LINE_RE = re.compile(
    r"(?m)^(?:\d{4}-\d{2}-\d{2}|(?:ERROR|WARN|INFO|DEBUG|CRITICAL|FATAL)\s|\[\w+\]\s)",
)

_SINGLE_KEYWORDS = [
    r"what\s+is\s+the",
    r"what\s+is\s+(?:a|an)\s+",
    r"how\s+do\s+i",
    r"how\s+does",
    r"summarize|summarise",
    r"what\s+(?:are\s+)?the\s+steps",
    r"who\s+(?:do\s+i\s+)?contact",
    r"where\s+(?:do\s+i\s+)?find",
    r"when\s+(?:does|is|will)",
]

_COMPILED_MULTI = [re.compile(p, re.I) for p in _MULTI_KEYWORDS]
_COMPILED_SINGLE = [re.compile(p, re.I) for p in _SINGLE_KEYWORDS]


# ── Result model ─────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """Outcome of the routing step."""
    mode: str           # "single" | "multi"
    confidence: str     # "high" (heuristic) | "medium" (llm-confirmed)
    reasons: list       # human-readable bullet points


# ── Router ──────────────────────────────────────────────────────────────────

class RequestRouter:
    """
    Routes incoming user requests to single-agent or multi-agent mode.

    Parameters
    ----------
    llm:
        Optional LLM for the fall-back classification pass.
        If ``None``, only heuristics are used.
    trace_id:
        Propagated trace ID for log correlation.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._logger = CopilotLogger("router")
        if trace_id:
            self._logger.bind_trace(trace_id)

    def route(self, user_input: str) -> RoutingDecision:
        """
        Analyse *user_input* and return a :class:`RoutingDecision`.
        """
        self._logger.step("router", "Analysing input", f"{len(user_input)} chars")

        # Pass 1 — heuristics
        multi_reasons, single_reasons = self._heuristic_pass(user_input)

        # If heuristics are decisive, return immediately
        if multi_reasons:
            decision = RoutingDecision(
                mode="multi",
                confidence="high",
                reasons=multi_reasons,
            )
            self._logger.step("router", "Decision", f"MULTI — {multi_reasons[0]}")
            return decision

        if single_reasons or len(user_input) < _MULTI_LENGTH_THRESHOLD:
            decision = RoutingDecision(
                mode="single",
                confidence="high",
                reasons=single_reasons or ["short / simple question"],
            )
            self._logger.step("router", "Decision", "SINGLE — simple query")
            return decision

        # Pass 2 — LLM disambiguation (ambiguous inputs)
        if self._llm is not None:
            return self._llm_pass(user_input)

        # Default: single (conservative when no LLM available)
        return RoutingDecision(mode="single", confidence="medium", reasons=["default fallback"])

    # ── private helpers ───────────────────────────────────────────────────────

    def _heuristic_pass(self, text: str):
        multi_reasons = []
        single_reasons = []

        # Length check
        if len(text) > _MULTI_LENGTH_THRESHOLD:
            multi_reasons.append(f"long input ({len(text)} chars > {_MULTI_LENGTH_THRESHOLD})")

        # Log-paste detection
        if _LOG_LINE_RE.search(text):
            multi_reasons.append("log lines detected in input")

        # Keyword checks
        for pattern in _COMPILED_MULTI:
            if pattern.search(text):
                multi_reasons.append(f"keyword match: '{pattern.pattern}'")
                break   # one reason is enough

        for pattern in _COMPILED_SINGLE:
            if pattern.search(text):
                single_reasons.append(f"keyword match: '{pattern.pattern}'")
                break

        return multi_reasons, single_reasons

    def _llm_pass(self, text: str) -> RoutingDecision:
        """Ask the LLM to classify the request (Single or Multi-agent)."""
        self._logger.step("router", "LLM disambiguation pass")
        system_prompt = (
            "You are a request classifier for a corporate IT assistant.\n"
            "Decide whether to use SINGLE-AGENT or MULTI-AGENT mode.\n\n"
            "SINGLE-AGENT: simple, low-risk policy questions answerable from a single retrieval.\n"
            "MULTI-AGENT: complex incidents, multiple systems, high-risk requests (disable MFA, "
            "ignore policy), long noisy logs, or requests requiring cross-checking multiple policies.\n\n"
            "Reply with EXACTLY one word: SINGLE or MULTI — nothing else."
        )
        try:
            response = self._llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=text[:1500])]
            )
            verdict = response.content.strip().upper()
            if "MULTI" in verdict:
                return RoutingDecision(
                    mode="multi",
                    confidence="medium",
                    reasons=["LLM classified as multi-agent"],
                )
        except Exception as exc:
            self._logger.warning(f"LLM routing failed, defaulting to single: {exc}")

        return RoutingDecision(
            mode="single",
            confidence="medium",
            reasons=["LLM classified as single-agent"],
        )
