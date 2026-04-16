"""
Unit tests for the Incident Analyzer — no LLM or network required.

Run with:  pytest tests/test_incident_analyzer.py -v
"""

import pytest
from src.tools.incident_analyzer import IncidentAnalyzer, IncidentReport


@pytest.fixture
def analyzer():
    return IncidentAnalyzer()


# ── Severity assignment ───────────────────────────────────────────────────────

def test_p1_severity_200_users(analyzer):
    text = "VPN failing for 200 users since this morning."
    report = analyzer.analyze(text)
    assert report.severity == "P1"
    assert report.affected_users == 200


def test_p2_severity_25_users(analyzer):
    text = "About 25 users are affected by the authentication issue."
    report = analyzer.analyze(text)
    assert report.severity == "P2"
    assert report.affected_users == 25


def test_p3_severity_5_users(analyzer):
    text = "5 people cannot connect to VPN today."
    report = analyzer.analyze(text)
    assert report.severity == "P3"
    assert report.affected_users == 5


def test_p4_severity_single_user(analyzer):
    text = "1 user says their account is locked."
    report = analyzer.analyze(text)
    assert report.severity == "P4"
    assert report.affected_users == 1


def test_severity_heuristic_vpn_no_count(analyzer):
    """Without user count, VPN systems push to at least P2."""
    text = "VPN gateway is unreachable, certificate error."
    report = analyzer.analyze(text)
    assert report.severity in ("P1", "P2")


# ── System extraction ─────────────────────────────────────────────────────────

def test_systems_extracted_vpn(analyzer):
    report = analyzer.analyze("Cisco AnyConnect VPN failing, MFA via Duo timeout.")
    assert "vpn" in report.systems
    assert "anyconnect" in report.systems
    assert "duo" in report.systems


def test_systems_extracted_azure_ad(analyzer):
    report = analyzer.analyze("Azure AD sign-in failures on Office 365.")
    assert "azure ad" in report.systems
    assert "o365" in report.systems or "office 365" in report.systems


# ── PII detection ─────────────────────────────────────────────────────────────

def test_pii_email_detected(analyzer):
    text = "User john.doe@company.com cannot log in."
    report = analyzer.analyze(text)
    assert report.pii_detected is True


def test_pii_ip_detected(analyzer):
    text = "Requests from 192.168.1.50 are being rejected."
    report = analyzer.analyze(text)
    assert report.pii_detected is True


def test_no_pii_clean_log(analyzer):
    text = "VPN gateway reported TLS timeout. 50 users affected."
    report = analyzer.analyze(text)
    assert report.pii_detected is False


# ── Timestamp extraction ──────────────────────────────────────────────────────

def test_timestamp_extracted(analyzer):
    text = "Error at 2025-04-16T09:15:33Z in vpn-gw01."
    report = analyzer.analyze(text)
    assert report.onset_time == "2025-04-16T09:15:33Z"


def test_no_timestamp(analyzer):
    text = "VPN is down, no idea when it started."
    report = analyzer.analyze(text)
    assert report.onset_time is None


# ── Suggested owner ───────────────────────────────────────────────────────────

def test_owner_network_ops_for_vpn(analyzer):
    text = "VPN and Cisco ASA are down for 100 users."
    report = analyzer.analyze(text)
    assert report.suggested_owner == "network-ops"


def test_owner_iam_for_ad(analyzer):
    text = "25 users cannot authenticate — Active Directory not responding."
    report = analyzer.analyze(text)
    assert report.suggested_owner == "iam-team"


# ── Ticket summary ────────────────────────────────────────────────────────────

def test_ticket_summary_contains_severity(analyzer):
    text = "200 users cannot use VPN. Certificate error in logs."
    report = analyzer.analyze(text)
    summary = report.to_ticket_summary()
    assert "P1" in summary
    assert "200 users" in summary


def test_pii_warning_in_summary(analyzer):
    text = "User john@company.com (200 affected) getting VPN error."
    report = analyzer.analyze(text)
    summary = report.to_ticket_summary()
    assert "PII" in summary.upper()


# ── PII redaction ─────────────────────────────────────────────────────────────

def test_redact_email(analyzer):
    text = "Contact user@company.com about the VPN issue."
    redacted = analyzer.redact_pii(text)
    assert "user@company.com" not in redacted
    assert "[EMAIL REDACTED]" in redacted


def test_redact_ip(analyzer):
    text = "Request from 10.0.0.5 was blocked."
    redacted = analyzer.redact_pii(text)
    assert "10.0.0.5" not in redacted
    assert "[IP REDACTED]" in redacted
