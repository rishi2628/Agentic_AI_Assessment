"""
Single-Agent — handles straightforward, low-risk policy questions.

Architecture
------------
Uses a LangGraph ReAct agent (``create_react_agent``) with one tool:
``retrieve_policy``.  The tool fetches relevant chunks from the ChromaDB
vector store and returns them with citations.

Guardrails
----------
* Input is checked BEFORE invoking the agent (prompt injection / disallowed advice).
* Response is checked AFTER the agent returns (PII in output, empty response).

Memory
------
Conversation history from ``SessionMemory`` is prepended as a system context
message so the agent can answer follow-up questions coherently.

Step limits
-----------
LangGraph enforces ``recursion_limit`` (defaults to ``MAX_STEPS`` from settings)
to prevent runaway tool-call loops.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

import config.settings as settings
from src.guardrails.safety_checker import SafetyChecker, GuardrailResult
from src.memory.session_memory import SessionMemory
from src.tools.policy_retriever import PolicyRetriever
from src.utils.logger import CopilotLogger


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the ACME Policy & Incident Copilot — an internal IT assistant.

Your role (SINGLE-AGENT mode):
- Answer employee questions about ACME IT policies and procedures.
- Only use the retrieve_policy tool to look up information. Do NOT make up answers.
- Provide the answer and ALWAYS include the source citation returned by the tool.
- Be concise. If the question cannot be answered from policy documents, say so clearly.

Constraints:
- Never advise disabling MFA or bypassing security controls.
- Never share or request credentials.
- If you are unsure, recommend contacting helpdesk@company.com.
"""


# ── Agent class ───────────────────────────────────────────────────────────────

class SingleAgent:
    """
    ReAct agent for straightforward policy questions.

    Parameters
    ----------
    llm:          Configured chat model.
    retriever:    Initialised ``PolicyRetriever`` instance.
    memory:       Shared ``SessionMemory`` for this user session.
    trace_id:     Propagated trace ID for log correlation.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: PolicyRetriever,
        memory: Optional[SessionMemory] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._memory = memory or SessionMemory()
        self._safety = SafetyChecker()
        self._logger = CopilotLogger("single_agent", trace_id)
        self._agent = self._build_agent()

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> dict:
        """
        Process a single-agent request.

        Returns
        -------
        dict with keys:
            ``response``         — final answer string
            ``guardrail_result`` — :class:`GuardrailResult`
            ``citations``        — list of source strings
            ``trace_id``         — for log correlation
            ``steps``            — number of tool-call steps taken
        """
        trace_id = self._logger.trace_id

        # ── 1. Input guardrail ─────────────────────────────────────────────────
        input_check = self._safety.check_input(user_input)
        if input_check.blocked:
            self._logger.warning(
                f"Input blocked by guardrail: {input_check.category}"
            )
            return self._blocked_response(input_check, trace_id)

        self._logger.step("single_agent", "Input passed guardrails")

        # ── 2. Build message list (system + history + user) ────────────────────
        messages = [SystemMessage(content=_SYSTEM_PROMPT)]
        messages += self._memory.to_langchain_messages()
        messages.append(HumanMessage(content=user_input))

        # ── 3. Invoke ReAct agent ──────────────────────────────────────────────
        self._logger.step("single_agent", "Invoking ReAct agent")
        try:
            result = self._agent.invoke(
                {"messages": messages},
                config={"recursion_limit": settings.MAX_STEPS},
            )
        except Exception as exc:
            self._logger.error(f"Agent invocation failed: {exc}")
            return {
                "response": (
                    "I encountered an error while processing your request.  "
                    "Please contact helpdesk@company.com."
                ),
                "guardrail_result": GuardrailResult(passed=True, category="ok"),
                "citations": [],
                "trace_id": trace_id,
                "steps": 0,
            }

        # ── 4. Extract final answer ────────────────────────────────────────────
        final_messages = result.get("messages", [])
        steps = len([m for m in final_messages if hasattr(m, "tool_calls")])
        raw_answer = ""
        for msg in reversed(final_messages):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                raw_answer = msg.content
                break

        # ── 5. Output guardrail ────────────────────────────────────────────────
        safe_answer, output_check = self._safety.safe_response(raw_answer)
        self._logger.step("single_agent", "Response passed guardrails")

        # ── 6. Update memory ───────────────────────────────────────────────────
        self._memory.add_user(user_input)
        self._memory.add_assistant(safe_answer)

        return {
            "response": safe_answer,
            "guardrail_result": output_check,
            "citations": self._extract_citations(safe_answer),
            "trace_id": trace_id,
            "steps": steps,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _build_agent(self):
        """Build the LangGraph ReAct agent with the policy-retrieval tool."""
        retriever = self._retriever

        @tool
        def retrieve_policy(query: str) -> str:
            """
            Retrieve the most relevant sections from ACME policy documents.
            Use this for every policy-related question.
            Always cite the source returned.

            Args:
                query: The search query describing what policy information is needed.
            """
            docs = retriever.retrieve(query, k=4)
            if not docs:
                return "No relevant policy information found for that query."
            return retriever.format_context(docs)

        from langgraph.prebuilt import create_react_agent

        return create_react_agent(model=self._llm, tools=[retrieve_policy])

    @staticmethod
    def _extract_citations(text: str) -> list:
        """Pull [Source N: filename.txt vX.Y] style citations from text."""
        import re
        return re.findall(r"\[Source\s*\d+:[^\]]+\]", text)

    @staticmethod
    def _blocked_response(result: GuardrailResult, trace_id: str) -> dict:
        return {
            "response": f"⛔ {result.reason}",
            "guardrail_result": result,
            "citations": [],
            "trace_id": trace_id,
            "steps": 0,
        }
