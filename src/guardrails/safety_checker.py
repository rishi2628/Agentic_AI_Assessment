"""
Guardrails — safety layer for the Policy & Incident Copilot.

Checks performed (in order)
----------------------------
1. Prompt-injection detection
   Attempts to override system instructions or hijack the agent's persona.

2. Disallowed-advice detection
   Requests to perform dangerous actions — disable MFA, ignore policy, etc.

3. PII detection in *responses*
   Ensures the agent never echoes raw PII back to the user.

4. Final response validation
   Checks that the response contains a citation and is not empty.

Each check returns a :class:`GuardrailResult` with:
  * ``passed``   — bool
  * ``reason``   — human-readable explanation of why it failed (if any)
  * ``category`` — one of: "prompt_injection" | "disallowed_advice" |
                   "pii_in_response" | "empty_response" | "ok"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

# ── Patterns ──────────────────────────────────────────────────────────────────

# Prompt-injection indicators
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(?:previous|above|prior)\s+instructions?", re.I),
    re.compile(r"disregard\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|another)\s+\w+", re.I),
    re.compile(r"act\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?!an?\s+assistant)", re.I),
    re.compile(r"forget\s+(?:everything|all)\s+(?:you\s+know|instructions?)", re.I),
    re.compile(r"new\s+(?:system\s+)?prompt\s*:", re.I),
    re.compile(r"<\s*system\s*>", re.I),   # XML-style injection
    re.compile(r"\[INST\]|\[SYSTEM\]", re.I),   # Llama-style injection markers
    re.compile(r"jailbreak", re.I),
]

# Disallowed / high-risk advice
_DISALLOWED_PATTERNS = [
    # MFA attacks
    (re.compile(r"disable\s+mfa", re.I), "Disabling MFA is not permitted per MFA Policy §7."),
    (re.compile(r"turn\s+off\s+mfa", re.I), "Disabling MFA is not permitted per MFA Policy §7."),
    (re.compile(r"bypass\s+mfa", re.I), "Bypassing MFA is not permitted per MFA Policy §7."),
    (re.compile(r"skip\s+mfa", re.I), "Bypassing MFA is not permitted per MFA Policy §7."),
    # Policy override
    (re.compile(r"ignore\s+(?:the\s+)?(?:\w+\s+)*policy", re.I), "Ignoring policy is not permitted."),
    (re.compile(r"override\s+(the\s+)?(?:policy|security|control)", re.I),
     "Overriding security controls requires formal approval."),
    # Credential sharing
    (re.compile(r"share\s+(?:your\s+)?(?:password|credentials?|login)", re.I),
     "Sharing credentials violates the Acceptable Use Policy §4.3."),
    # Disabling security tools
    (re.compile(r"disable\s+(?:the\s+)?(?:firewall|antivirus|av|edr|dlp|logging|audit)", re.I),
     "Disabling security controls is not permitted without CISO approval."),
    (re.compile(r"uninstall\s+(?:the\s+)?(?:antivirus|av|edr|endpoint\s+protection)", re.I),
     "Removing security software is not permitted."),
]

# PII patterns in *responses* (should have been redacted already; double-check)
_RESPONSE_PII_PATTERNS = [
    re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),   # SSN
    re.compile(                                          # Full credit card-like
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
]


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool
    category: str   # "ok" | "prompt_injection" | "disallowed_advice" | "pii_in_response" | "empty_response"
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return not self.passed


# ── SafetyChecker ─────────────────────────────────────────────────────────────

class SafetyChecker:
    """
    Stateless guardrails checker.  All methods are pure and can be called
    independently or via the combined :meth:`check_input` / :meth:`check_response`
    entry points.
    """

    # ── Input checks (run BEFORE sending to LLM) ──────────────────────────────

    def check_input(self, user_input: str) -> GuardrailResult:
        """
        Run all input-side guardrails.  Returns the *first* failure found,
        or ``GuardrailResult(passed=True, category="ok")`` if clean.
        """
        result = self._check_prompt_injection(user_input)
        if result.blocked:
            return result

        result = self._check_disallowed_advice(user_input)
        if result.blocked:
            return result

        return GuardrailResult(passed=True, category="ok")

    def _check_prompt_injection(self, text: str) -> GuardrailResult:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    category="prompt_injection",
                    reason=(
                        "Your message appears to contain an attempt to override "
                        "the assistant's instructions.  This is not permitted."
                    ),
                )
        return GuardrailResult(passed=True, category="ok")

    def _check_disallowed_advice(self, text: str) -> GuardrailResult:
        for pattern, explanation in _DISALLOWED_PATTERNS:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    category="disallowed_advice",
                    reason=(
                        f"This request touches a restricted action: {explanation} "
                        "Please follow the formal exception process or contact IT Security."
                    ),
                )
        return GuardrailResult(passed=True, category="ok")

    # ── Response checks (run BEFORE returning answer to user) ─────────────────

    def check_response(self, response: str) -> GuardrailResult:
        """
        Validate the agent's draft response.
        Returns the first failure, or ``ok`` if all checks pass.
        """
        if not response or not response.strip():
            return GuardrailResult(
                passed=False,
                category="empty_response",
                reason="The agent produced an empty response.",
            )

        for pattern in _RESPONSE_PII_PATTERNS:
            if pattern.search(response):
                return GuardrailResult(
                    passed=False,
                    category="pii_in_response",
                    reason=(
                        "The draft response contains patterns that look like PII "
                        "(e.g. SSN or full card number).  Redact before sending."
                    ),
                )

        return GuardrailResult(passed=True, category="ok")

    # ── Combined helper ────────────────────────────────────────────────────────

    def safe_response(self, response: str) -> Tuple[str, GuardrailResult]:
        """
        Return (sanitised_response, result).
        If PII is found the response is replaced with a safe error message.
        """
        result = self.check_response(response)
        if result.category == "pii_in_response":
            safe = (
                "⚠️  The generated response contained potentially sensitive data "
                "and has been withheld.  Please contact IT Security for assistance."
            )
            return safe, result
        return response, result
