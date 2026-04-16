"""
Incident Analyzer — structured first-pass triage of user-reported incidents.

This tool is called by the multi-agent *researcher* node.  It does NOT call the
LLM itself; it performs deterministic pattern-matching to extract structured
metadata from raw incident text.  The extracted metadata guides the planner's
subtask generation and sets the ticket severity.

Extracted fields
----------------
* severity        P1 / P2 / P3 / P4
* affected_users  integer (or None if unknown)
* systems         list of mentioned system names
* error_keywords  list of error tokens found in the text
* onset_time      raw timestamp string extracted from logs (if present)
* log_lines       list of individual log lines (if a log block is pasted)
* pii_detected    True if patterns matching PII are found (triggers redaction)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Patterns ─────────────────────────────────────────────────────────────────

_USER_COUNT_RE = re.compile(
    r"(\d[\d,]*)\s+(?:users?|employees?|staff|people|accounts?|machines?)",
    re.IGNORECASE,
)
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?",
)
_IP_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
# Intentionally broad: catch names in "user: John Smith" style log entries
_NAME_RE = re.compile(
    r"\b(?:user|employee|name|account)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
)

_KNOWN_SYSTEMS = [
    "vpn", "anyconnect", "duo", "mfa", "azure ad", "active directory", "ad",
    "office 365", "o365", "sharepoint", "onedrive", "aws", "rdp", "rdg",
    "github", "servicenow", "datadog", "pagerduty", "ldap", "ldaps",
    "cisco asa", "firewall", "dns", "proxy", "sso", "okta", "ping",
]

_ERROR_KEYWORDS = [
    "error", "fail", "failed", "failure", "timeout", "timed out", "refused",
    "unreachable", "denied", "certificate", "expired", "invalid", "exception",
    "crash", "down", "outage", "disconnected", "cannot connect", "unable to",
    "authentication failed", "401", "403", "500", "502", "503", "504",
]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class IncidentReport:
    """Structured result of first-pass incident analysis."""

    raw_text: str
    severity: str = "P3"
    affected_users: Optional[int] = None
    systems: List[str] = field(default_factory=list)
    error_keywords: List[str] = field(default_factory=list)
    onset_time: Optional[str] = None
    log_lines: List[str] = field(default_factory=list)
    pii_detected: bool = False
    suggested_owner: str = "help-desk"

    def to_ticket_summary(self) -> str:
        """Return a formatted one-paragraph ticket summary."""
        users_str = (
            f"{self.affected_users} users" if self.affected_users else "unknown number of users"
        )
        systems_str = ", ".join(self.systems) if self.systems else "unspecified system"
        errors_str = "; ".join(self.error_keywords[:5]) if self.error_keywords else "no specific errors identified"
        pii_warn = "\n⚠️  PII MAY BE PRESENT — review before sharing externally." if self.pii_detected else ""

        return (
            f"[{self.severity}] Incident affecting {users_str} on {systems_str}.\n"
            f"Errors detected: {errors_str}.\n"
            f"Onset: {self.onset_time or 'not specified'}.\n"
            f"Suggested owner: {self.suggested_owner}."
            f"{pii_warn}"
        )


# ── Analyzer ─────────────────────────────────────────────────────────────────

class IncidentAnalyzer:
    """
    Stateless analyser — all public methods are pure functions that take
    the incident text and return structured results.
    """

    # ── public entry point ────────────────────────────────────────────────────

    def analyze(self, text: str) -> IncidentReport:
        """
        Run all extraction passes and return a fully populated
        :class:`IncidentReport`.
        """
        report = IncidentReport(raw_text=text)
        report.affected_users = self._extract_user_count(text)
        report.onset_time = self._extract_timestamp(text)
        report.systems = self._extract_systems(text)
        report.error_keywords = self._extract_error_keywords(text)
        report.log_lines = self._extract_log_lines(text)
        report.pii_detected = self._detect_pii(text)
        report.severity = self._assign_severity(
            report.affected_users, report.systems, report.error_keywords
        )
        report.suggested_owner = self._suggest_owner(report.systems, report.severity)
        return report

    # ── extraction helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_user_count(text: str) -> Optional[int]:
        matches = _USER_COUNT_RE.findall(text)
        if not matches:
            return None
        counts = [int(m.replace(",", "")) for m in matches]
        return max(counts)    # most severe count wins

    @staticmethod
    def _extract_timestamp(text: str) -> Optional[str]:
        match = _TIMESTAMP_RE.search(text)
        return match.group(0) if match else None

    @staticmethod
    def _extract_systems(text: str) -> List[str]:
        lower = text.lower()
        found = [sys for sys in _KNOWN_SYSTEMS if sys in lower]
        # deduplicate while preserving order
        seen: set = set()
        result = []
        for s in found:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    @staticmethod
    def _extract_error_keywords(text: str) -> List[str]:
        lower = text.lower()
        found = [kw for kw in _ERROR_KEYWORDS if kw in lower]
        seen: set = set()
        result = []
        for k in found:
            if k not in seen:
                seen.add(k)
                result.append(k)
        return result[:10]   # cap at 10 to keep the ticket readable

    @staticmethod
    def _extract_log_lines(text: str) -> List[str]:
        """
        Heuristic: a log line starts with a timestamp or known log levels.
        """
        log_pattern = re.compile(
            r"^(?:\d{4}-\d{2}-\d{2}|\[\w+\]|\w+\s+\w+:\s+).+",
            re.MULTILINE,
        )
        return log_pattern.findall(text)[:20]  # return up to 20 lines

    @staticmethod
    def _detect_pii(text: str) -> bool:
        """Return True if email addresses, IP addresses, or person names are found."""
        if _EMAIL_RE.search(text):
            return True
        if _IP_RE.search(text):
            return True
        if _NAME_RE.search(text):
            return True
        return False

    @staticmethod
    def _assign_severity(
        user_count: Optional[int],
        systems: List[str],
        errors: List[str],
    ) -> str:
        if user_count is not None:
            if user_count > 50:
                return "P1"
            if user_count >= 10:
                return "P2"
            if user_count >= 2:
                return "P3"
            return "P4"

        # No user count — use heuristics
        critical_systems = {"vpn", "active directory", "ad", "azure ad", "mfa", "duo"}
        if any(s in critical_systems for s in systems):
            return "P2"
        if errors:
            return "P3"
        return "P4"

    @staticmethod
    def _suggest_owner(systems: List[str], severity: str) -> str:
        if any(s in {"vpn", "anyconnect", "cisco asa", "firewall", "dns"} for s in systems):
            return "network-ops"
        if any(s in {"azure ad", "active directory", "ad", "ldap", "mfa", "duo"} for s in systems):
            return "iam-team"
        if any(s in {"aws"} for s in systems):
            return "cloud-ops"
        if severity in {"P1", "P2"}:
            return "sre-oncall"
        return "help-desk"


    def redact_pii(self, text: str) -> str:
        """
        Redact email addresses, IPs, and person-name patterns from *text*.
        Should be applied before any log content is surfaced in a response.
        """
        text = _EMAIL_RE.sub("[EMAIL REDACTED]", text)
        text = _IP_RE.sub("[IP REDACTED]", text)
        text = _NAME_RE.sub(r"\1[NAME REDACTED]", text)
        return text
