"""
Multi-Agent Pipeline — handles complex, high-risk incidents and multi-step requests.

Architecture (LangGraph StateGraph)
-------------------------------------

    ┌─────────────────────────────────────────────────────────────────┐
    │                    CopilotState (shared)                        │
    └─────────────────────────────────────────────────────────────────┘
           │
        START
           │
      ┌────▼────┐
      │ planner │  LLM breaks the request into ≤5 descriptive subtasks
      └────┬────┘
           │
     ┌─────▼──────┐
     │ researcher │  For each subtask: RAG retrieval + incident analysis
     └─────┬──────┘
           │
    ┌──────▼──────┐
    │ draft_writer│  LLM synthesises a structured ticket / answer
    └──────┬──────┘
           │
      ┌────▼────┐
      │  critic │  LLM verifies draft against retrieved sources
      └────┬────┘
           │ (max MAX_REVISIONS loops back to researcher)
           │
      ┌────▼────┐
      │ safety  │  Guardrails: PII + disallowed-advice checks
      └────┬────┘
           │
          END

State fields
------------
* ``user_input``       — original user message
* ``plan``             — list of subtask strings (set by planner)
* ``retrieved_docs``   — list of retrieved doc dicts from all subtasks
* ``incident_report``  — structured IncidentReport (if incident text present)
* ``draft_response``   — current draft answer
* ``critic_feedback``  — reviewer notes
* ``is_verified``      — whether critic approved the draft
* ``safety_result``    — GuardrailResult from the safety node
* ``final_response``   — output shown to the user
* ``trace_id``         — UUID for log correlation
* ``steps``            — counter to enforce MAX_STEPS
* ``revision_count``   — loops through researcher→draft→critic
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import config.settings as settings
from src.guardrails.safety_checker import GuardrailResult, SafetyChecker
from src.tools.incident_analyzer import IncidentAnalyzer, IncidentReport
from src.tools.policy_retriever import PolicyRetriever
from src.utils.logger import CopilotLogger


# ── Standalone formatting helper (no ChromaDB connection needed) ──────────────

def _format_context(docs: List[Dict[str, Any]]) -> str:
    """Format a list of retrieved doc dicts into a numbered citation block."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        citation = f"[Source {i}: {doc.get('source', 'unknown')} {doc.get('version', '')}]"
        lines.append(f"{citation}\n{doc['text']}\n")
    return "\n---\n".join(lines)


# ── Shared state ──────────────────────────────────────────────────────────────

class CopilotState(TypedDict):
    user_input: str
    plan: List[str]
    retrieved_docs: List[Dict[str, Any]]
    incident_report: Optional[Dict]          # serialised IncidentReport
    draft_response: str
    critic_feedback: str
    is_verified: bool
    safety_result: Optional[Dict]            # serialised GuardrailResult
    final_response: str
    trace_id: str
    steps: int
    revision_count: int


# ── System prompts ────────────────────────────────────────────────────────────

_PLANNER_PROMPT = """You are a planning agent for a corporate IT assistant.

Your task: Break the user's request into at most 5 concise subtasks.
Each subtask must be a specific search question for a policy/procedure lookup
or a specific aspect of the incident to investigate.

RULES:
- Output ONLY a numbered list: 1. ... 2. ... etc.
- No preamble, no explanation.
- If the input contains an incident, include subtasks for: blast-radius, policy lookup,
  triage steps, and recommended owner.
"""

_DRAFT_WRITER_PROMPT = """You are a drafting agent for a corporate IT assistant.

Write a structured response based on the plan and retrieved policy context provided.

For INCIDENT requests, use this ticket format:
  [SEVERITY] [SYSTEM] — <brief title>
  Affected users: <N>
  Onset: <timestamp or unknown>
  Triage steps performed: <list>
  Recommended owner: <team>
  Policy references: <citations>
  Immediate action: <steps>
  Escalation path: <if P1/P2>

For POLICY questions, use:
  Answer: <concise answer>
  Effective date / version: <from source>
  Source: <citation>
  Next steps (if applicable): <>

IMPORTANT: Do NOT include raw PII (emails, SSNs, full names from logs) in your output.
Always cite your sources using the [Source N: filename vX.Y] format.
"""

_CRITIC_PROMPT = """You are a critic/verifier agent for a corporate IT assistant.

Review the DRAFT RESPONSE against the RETRIEVED CONTEXT and provide your verdict.

CHECKLIST:
1. Is the answer factually supported by the retrieved policy context?
2. Are citations present and accurate?
3. Is PII present that should be redacted?
4. Are any disallowed actions suggested (disable MFA, ignore policy)?
5. For incidents: is severity correctly assigned per the Incident Response Procedure?

OUTPUT FORMAT:
VERDICT: APPROVED | NEEDS_REVISION
ISSUES: <numbered list, or "None">
"""


# ── Node functions ────────────────────────────────────────────────────────────

def planner_node(state: CopilotState, llm: BaseChatModel, logger: CopilotLogger) -> CopilotState:
    logger.step("planner", "Decomposing request")
    response = llm.invoke([
        SystemMessage(content=_PLANNER_PROMPT),
        HumanMessage(content=state["user_input"][:3000]),
    ])

    # Parse numbered list into clean strings
    import re
    lines = response.content.strip().splitlines()
    plan = []
    for line in lines:
        item = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if item:
            plan.append(item)

    plan = plan[:5] or ["Retrieve general policy information"]
    logger.step("planner", f"Plan ({len(plan)} subtasks)", str(plan))
    return {**state, "plan": plan, "steps": state["steps"] + 1}


def researcher_node(state: CopilotState, retriever: PolicyRetriever, logger: CopilotLogger) -> CopilotState:
    logger.step("researcher", "Retrieving policy documents")
    analyzer = IncidentAnalyzer()
    all_docs: List[Dict] = []

    for subtask in state["plan"]:
        docs = retriever.retrieve(subtask, k=3)
        for d in docs:
            d["subtask"] = subtask
        all_docs.extend(docs)

    # Deduplicate by (source, chunk text[:50])
    seen: set = set()
    unique_docs: List[Dict] = []
    for d in all_docs:
        key = (d["source"], d["text"][:50])
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    logger.step("researcher", f"Retrieved {len(unique_docs)} unique chunks")

    # Run incident analysis on the original input
    report = analyzer.analyze(state["user_input"])
    incident_dict = {
        "severity": report.severity,
        "affected_users": report.affected_users,
        "systems": report.systems,
        "error_keywords": report.error_keywords,
        "onset_time": report.onset_time,
        "pii_detected": report.pii_detected,
        "suggested_owner": report.suggested_owner,
        "ticket_summary": report.to_ticket_summary(),
    }

    return {**state, "retrieved_docs": unique_docs, "incident_report": incident_dict, "steps": state["steps"] + 1}


def draft_writer_node(state: CopilotState, llm: BaseChatModel, logger: CopilotLogger) -> CopilotState:
    logger.step("draft_writer", "Generating draft response")
    context = _format_context(state["retrieved_docs"][:8])

    incident = state.get("incident_report") or {}
    incident_ctx = ""
    if incident:
        incident_ctx = (
            f"\n\nINCIDENT ANALYSIS:\n{incident.get('ticket_summary', '')}"
        )

    history = ""
    if state.get("critic_feedback"):
        history = f"\n\nPREVIOUS CRITIC FEEDBACK (address these):\n{state['critic_feedback']}"

    prompt = (
        f"PLAN:\n" + "\n".join(f"- {s}" for s in state["plan"]) +
        f"\n\nRETRIEVED POLICY CONTEXT:\n{context}" +
        incident_ctx +
        history
    )

    response = llm.invoke([
        SystemMessage(content=_DRAFT_WRITER_PROMPT),
        HumanMessage(content=prompt),
    ])

    logger.step("draft_writer", "Draft complete")
    return {**state, "draft_response": response.content, "steps": state["steps"] + 1}


def critic_node(state: CopilotState, llm: BaseChatModel, logger: CopilotLogger) -> CopilotState:
    logger.step("critic", "Verifying draft response")
    context = _format_context(state["retrieved_docs"][:5])

    prompt = (
        f"DRAFT RESPONSE:\n{state['draft_response']}\n\n"
        f"RETRIEVED CONTEXT:\n{context}"
    )

    response = llm.invoke([
        SystemMessage(content=_CRITIC_PROMPT),
        HumanMessage(content=prompt),
    ])

    verdict_text = response.content.upper()
    is_verified = "APPROVED" in verdict_text and "NEEDS_REVISION" not in verdict_text

    logger.step(
        "critic",
        "Verdict",
        "APPROVED" if is_verified else "NEEDS_REVISION",
    )
    return {
        **state,
        "critic_feedback": response.content,
        "is_verified": is_verified,
        "revision_count": state["revision_count"] + 1,
        "steps": state["steps"] + 1,
    }


def safety_node(state: CopilotState, logger: CopilotLogger) -> CopilotState:
    logger.step("safety", "Running guardrail checks")
    checker = SafetyChecker()

    # Check for disallowed advice in the draft
    input_result = checker.check_input(state["draft_response"])
    if input_result.blocked:
        logger.warning(f"Draft blocked by guardrail: {input_result.category}")
        blocked_msg = (
            f"⛔ The generated response was blocked by the safety filter.\n"
            f"Reason: {input_result.reason}\n\n"
            "Please rephrase your request or contact IT Security directly at "
            "soc@company.com."
        )
        return {
            **state,
            "final_response": blocked_msg,
            "safety_result": {"passed": False, "category": input_result.category, "reason": input_result.reason},
            "steps": state["steps"] + 1,
        }

    safe_response, output_result = checker.safe_response(state["draft_response"])
    logger.step("safety", "Guardrails passed")
    return {
        **state,
        "final_response": safe_response,
        "safety_result": {"passed": output_result.passed, "category": output_result.category, "reason": output_result.reason},
        "steps": state["steps"] + 1,
    }


# ── Conditional edge helpers ──────────────────────────────────────────────────

def should_revise_or_proceed(
    state: CopilotState,
) -> Literal["researcher", "safety"]:
    """After critic: loop back for revision, or proceed to safety check."""
    max_rev = settings.MAX_REVISIONS
    if not state["is_verified"] and state["revision_count"] < max_rev:
        return "researcher"
    return "safety"


def check_step_limit(state: CopilotState) -> Literal["planner", "__end__"]:
    """Guard against runaway graphs."""
    if state["steps"] >= settings.MAX_STEPS:
        return "__end__"
    return "planner"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_multi_agent_graph(
    llm: BaseChatModel,
    retriever: PolicyRetriever,
    logger: CopilotLogger,
):
    """
    Compile and return the LangGraph ``CompiledGraph`` for multi-agent mode.
    Nodes are partially applied with their dependencies so the graph
    only needs ``state`` as the node function argument.
    """
    graph = StateGraph(CopilotState)

    graph.add_node("planner",      lambda s: planner_node(s, llm, logger))
    graph.add_node("researcher",   lambda s: researcher_node(s, retriever, logger))
    graph.add_node("draft_writer", lambda s: draft_writer_node(s, llm, logger))
    graph.add_node("critic",       lambda s: critic_node(s, llm, logger))
    graph.add_node("safety",       lambda s: safety_node(s, logger))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "draft_writer")
    graph.add_edge("draft_writer", "critic")
    graph.add_conditional_edges(
        "critic",
        should_revise_or_proceed,
        {"researcher": "researcher", "safety": "safety"},
    )
    graph.add_edge("safety", END)

    return graph.compile()


# ── MultiAgent class ──────────────────────────────────────────────────────────

class MultiAgent:
    """
    High-level wrapper around the LangGraph multi-agent pipeline.

    Parameters
    ----------
    llm:        Configured chat model.
    retriever:  Initialised PolicyRetriever.
    trace_id:   Propagated trace ID for log correlation.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: PolicyRetriever,
        trace_id: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._safety = SafetyChecker()
        self._logger = CopilotLogger("multi_agent", trace_id)
        self._graph = build_multi_agent_graph(llm, retriever, self._logger)

    def run(self, user_input: str) -> dict:
        """
        Run the full multi-agent pipeline.

        Returns
        -------
        dict with keys:
            ``response``       — final answer / ticket
            ``severity``       — incident severity (or None for policy Q)
            ``citations``      — list of source citations
            ``trace_id``       — for log correlation
            ``steps``          — total pipeline steps executed
            ``plan``           — decomposed subtask list
            ``safety_result``  — dict with passed / category / reason
        """
        trace_id = self._logger.trace_id

        # Input guardrail before the pipeline
        input_check = self._safety.check_input(user_input)
        if input_check.blocked:
            self._logger.warning(f"Input blocked: {input_check.category}")
            return {
                "response": f"⛔ {input_check.reason}",
                "severity": None,
                "citations": [],
                "trace_id": trace_id,
                "steps": 0,
                "plan": [],
                "safety_result": {
                    "passed": False,
                    "category": input_check.category,
                    "reason": input_check.reason,
                },
            }

        initial_state: CopilotState = {
            "user_input": user_input,
            "plan": [],
            "retrieved_docs": [],
            "incident_report": None,
            "draft_response": "",
            "critic_feedback": "",
            "is_verified": False,
            "safety_result": None,
            "final_response": "",
            "trace_id": trace_id,
            "steps": 0,
            "revision_count": 0,
        }

        self._logger.step("multi_agent", "Starting pipeline")
        try:
            final_state = self._graph.invoke(
                initial_state,
                config={"recursion_limit": settings.MAX_STEPS + 10},  # graph overhead
            )
        except Exception as exc:
            self._logger.error(f"Multi-agent pipeline failed: {exc}")
            return {
                "response": (
                    "The incident triage pipeline encountered an error.  "
                    "Please escalate to the on-call engineer or contact soc@company.com."
                ),
                "severity": None,
                "citations": [],
                "trace_id": trace_id,
                "steps": 0,
                "plan": [],
                "safety_result": {"passed": True, "category": "ok", "reason": ""},
            }

        incident = final_state.get("incident_report") or {}
        return {
            "response": final_state["final_response"],
            "severity": incident.get("severity"),
            "citations": self._extract_citations(final_state["final_response"]),
            "trace_id": trace_id,
            "steps": final_state["steps"],
            "plan": final_state["plan"],
            "safety_result": final_state.get("safety_result") or {"passed": True, "category": "ok", "reason": ""},
        }

    @staticmethod
    def _extract_citations(text: str) -> list:
        import re
        return re.findall(r"\[Source\s*\d+:[^\]]+\]", text)
