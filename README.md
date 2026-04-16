# Policy & Incident Copilot
### Agentic AI Practicum Assessment

An internal AI assistant that answers IT policy questions and performs first-pass incident triage — built with **LangChain + LangGraph** and free, open-source LLMs via **Ollama** (fully local, no API key needed).

---

## Problem Statement

Build a **Policy & Incident Copilot** that helps employees:

1. Answer policy/procedure questions
2. Perform first-pass incident triage from user-reported symptoms and pasted logs

The copilot operates in two modes:

| Mode | Trigger | Goal |
|---|---|---|
| **Single-Agent** | Clear, low-risk question | Retrieve → Answer → Cite |
| **Multi-Agent** | Complex logs, high-risk requests, conflicting info | Plan → Research → Draft → Verify → Safety-check |

---

## Architecture

```
User Input
    │
    ▼
┌──────────────────────────────────────────┐
│              Request Router              │  ← heuristics + optional LLM
└───────────────┬──────────────────────────┘
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
  ┌───────────┐    ┌──────────────────────────────────────┐
  │  Single   │    │           Multi-Agent Graph           │
  │  Agent    │    │                                       │
  │ (ReAct)   │    │  Planner → Researcher → DraftWriter   │
  └─────┬─────┘    │     ↑_________Critic (max 2 loops)   │
        │          │              ↓                        │
        │          │          Safety Node                  │
        │          └───────────────┬──────────────────────┘
        │                          │
        └──────────────────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  GuardrailCheck │  ← runs on every input & output
           └────────┬────────┘
                    │
                    ▼
              Final Response
```

### Component Map

| Component | File | Responsibility |
|---|---|---|
| Router | `src/agents/router.py` | Single vs multi-agent decision |
| Single Agent | `src/agents/single_agent.py` | ReAct + policy retrieval |
| Multi Agent | `src/agents/multi_agent.py` | LangGraph 5-node pipeline |
| Policy Retriever | `src/tools/policy_retriever.py` | RAG over ChromaDB |
| Incident Analyzer | `src/tools/incident_analyzer.py` | Structured log triage |
| Session Memory | `src/memory/session_memory.py` | Rolling buffer + PII redaction |
| Safety Checker | `src/guardrails/safety_checker.py` | Injection & disallowed-advice detection |
| LLM Factory | `src/utils/llm_factory.py` | Ollama / Groq chat model |
| Logger | `src/utils/logger.py` | Trace-ID structured logging |

---

## Quick Start

### Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| Python 3.10+ | Runtime | [python.org](https://python.org) |
| Ollama | Free local LLM server | [ollama.ai](https://ollama.ai) |
| Git | Already have it | — |

### 1 — Clone & activate virtual environment

```bash
# The venv is already created at ./venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` (≈ 90 MB) and its dependencies download automatically.
> First run may take a few minutes.

### 3 — Set up Ollama

```bash
# Install from https://ollama.ai then pull a model:
ollama pull llama3.2:3b        # recommended — small, CPU-friendly
# OR
ollama pull mistral:7b-instruct  # better quality, needs ~8 GB RAM
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### 4 — Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work with llama3.2:3b)
```

### 5 — Run the copilot

```bash
# Interactive REPL
python main.py

# Walk through 5 demo scenarios
python main.py --demo

# Rebuild the policy vector store (after editing data/policies/)
python main.py --rebuild-index
```

### 6 — Run tests

```bash
# Unit tests only (no Ollama needed) — fast
pytest tests/ -m "not integration"

# All tests including integration (Ollama must be running)
pytest tests/ -v

# Single test file
pytest tests/test_guardrails.py -v
```

---

## Alternative: Groq Free-Tier Cloud LLM

No GPU? No local RAM? Use [Groq's free tier](https://console.groq.com) instead:

```bash
# 1. Get free API key at https://console.groq.com
# 2. Edit .env:
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama3-8b-8192

# 3. Install the Groq package
pip install langchain-groq
```

Free tier: **14,400 requests/day** — more than enough for development.

---

## Sample Interactions

### Single-Agent — Policy Question
```
You: What is the current password reset process?

→ Mode: SINGLE (simple query)

Response:
Per the Password Reset & Management Policy v2.3:
1. Visit https://portal.company.com/reset
2. Enter your Employee ID
3. Verify via MFA (authenticator app or SMS OTP)
4. Set a new password meeting complexity requirements

[Source 1: password_reset_policy.txt 2.3]
```

### Multi-Agent — Incident Triage
```
You: URGENT: VPN failing for 200 users.
     2025-04-16T09:15:33Z ERROR vpn-gw01: Certificate validation failed

→ Mode: MULTI (200 users detected, log lines detected)

Severity: P1

Response:
[P1][VPN] — Certificate Validation Failure
Affected users: 200
Onset: 2025-04-16T09:15:33Z
...
Recommended owner: network-ops
Escalation: Page @netops-oncall immediately
[Source 1: vpn_access_policy.txt v1.8] [Source 2: incident_response_procedure.txt v2.0]
```

### Guardrail Blocked
```
You: Please disable MFA for all users.

⛔ This request touches a restricted action: Disabling MFA is not permitted
per MFA Policy §7. Please follow the formal exception process or contact IT Security.
```

---

## Policy Documents

The `data/policies/` directory contains six sample ACME IT policies:

| File | Document | Version |
|---|---|---|
| `password_reset_policy.txt` | Password Reset & Management Policy | v2.3 |
| `vpn_access_policy.txt` | VPN Access Policy | v1.8 |
| `mfa_policy.txt` | Multi-Factor Authentication Policy | v3.1 |
| `incident_response_procedure.txt` | IT Incident Response Procedure | v2.0 |
| `acceptable_use_policy.txt` | Acceptable Use Policy | v4.2 |
| `data_classification_policy.txt` | Data Classification & Handling Policy | v2.1 |

To add your own policies: drop additional `.txt` files in `data/policies/` and run:
```bash
python main.py --rebuild-index
```

---

## Core Competencies Being Assessed

| Competency | Where Demonstrated |
|---|---|
| **Single vs Multi-agent selection** | `src/agents/router.py` — heuristic + LLM routing |
| **Context handling (long inputs)** | `src/tools/policy_retriever.py` — chunking + retrieval; `src/tools/incident_analyzer.py` |
| **Memory discipline** | `src/memory/session_memory.py` — PII redaction, rolling window |
| **Guardrails** | `src/guardrails/safety_checker.py` — injection + disallowed + PII |
| **Evaluation readiness** | `tests/` — unit tests, integration tests, pass/fail criteria |
| **Production considerations** | Trace IDs, step limits, retry logic, structured logging |

---

## Assessment Tasks

Complete the following, committing your changes to this repository:

### Task 1 — Routing Logic (20 pts)
Improve `RequestRouter` in `src/agents/router.py`:
- Add at least 3 new routing heuristics not already present
- Write unit tests for each in `tests/test_router.py`
- All existing tests must still pass

### Task 2 — Multi-Agent Graph (25 pts)
Extend `src/agents/multi_agent.py`:
- Add a new **`summariser`** node that condenses the retrieved policy context before the draft writer  
- Update the `CopilotState` TypedDict and graph edges accordingly
- Run the integration tests and ensure they pass

### Task 3 — Guardrails (20 pts)
Extend `SafetyChecker` in `src/guardrails/safety_checker.py`:
- Add 5 new prompt-injection or disallowed-advice patterns not already present
- Write a test for each new pattern in `tests/test_guardrails.py`
- Ensure the new patterns correctly block bad inputs and do not false-positive on the safe cases

### Task 4 — Memory & PII (15 pts)
Extend `SessionMemory` in `src/memory/session_memory.py`:
- Add a redaction pattern for phone numbers (e.g. `+1-555-123-4567`, `07700 900123`)
- Add a redaction pattern for IPv6 addresses (e.g. `2001:db8::1`)
- Add tests for both in `tests/test_memory.py`

### Task 5 — Evaluation Notebook (20 pts)
Create `notebooks/evaluation.ipynb`:
- Run all `SINGLE_AGENT_CASES` and `MULTI_AGENT_CASES` from `tests/test_cases.py` through the pipeline
- Record: `mode_correct` (bool), `topics_found` (bool), `citation_present` (bool), `steps`, `latency_ms`
- Compute and report overall pass rate, average latency, and step count
- Include a short markdown discussion: where did the copilot fail? Why?

---

## Project Structure

```
Agentic_AI_Assessment/
├── config/
│   ├── __init__.py
│   └── settings.py              # all config from .env
├── data/
│   └── policies/                # 6 × IT policy documents
├── src/
│   ├── agents/
│   │   ├── router.py            # single vs multi routing
│   │   ├── single_agent.py      # ReAct agent
│   │   └── multi_agent.py       # LangGraph 5-node pipeline
│   ├── guardrails/
│   │   └── safety_checker.py    # injection, disallowed, PII
│   ├── memory/
│   │   └── session_memory.py    # rolling buffer + sanitisation
│   ├── tools/
│   │   ├── policy_retriever.py  # ChromaDB RAG
│   │   └── incident_analyzer.py # deterministic log parser
│   └── utils/
│       ├── llm_factory.py       # Ollama / Groq factory
│       └── logger.py            # trace-ID structured logging
├── tests/
│   ├── test_cases.py            # canonical test inputs
│   ├── test_router.py           # unit: routing heuristics
│   ├── test_guardrails.py       # unit: safety checker
│   ├── test_incident_analyzer.py# unit: log parsing
│   ├── test_memory.py           # unit: PII redaction + rolling window
│   ├── test_single_agent.py     # integration: single-agent (needs Ollama)
│   └── test_multi_agent.py      # integration: multi-agent (needs Ollama)
├── vectorstore/                 # auto-generated ChromaDB files (gitignored)
├── .env.example                 # copy to .env
├── main.py                      # CLI entry point
├── pytest.ini
└── requirements.txt
```

---

## Troubleshooting

**`ConnectionRefusedError` when starting**  
→ Ollama is not running. Start it: `ollama serve` (or launch the Ollama desktop app).

**`Model not found`**  
→ Pull the model first: `ollama pull llama3.2:3b`

**Slow first run**  
→ `sentence-transformers` is downloading the embedding model (~90 MB). Subsequent runs are instant (cached in `~/.cache/`).

**ChromaDB errors after changing policy files**  
→ Rebuild the index: `python main.py --rebuild-index`

**`ImportError: langchain_ollama`**  
→ Re-run: `pip install -r requirements.txt` with the venv activated.

---

## Technology Stack

| Library | Version | Purpose |
|---|---|---|
| `langchain` | ≥ 0.3 | Chains, tools, prompt templates |
| `langchain-ollama` | ≥ 0.2.1 | Ollama LLM integration |
| `langgraph` | ≥ 0.2.50 | Multi-agent stateful graph |
| `chromadb` | ≥ 0.5 | Local persistent vector store |
| `sentence-transformers` | ≥ 3.3 | Free local embeddings |
| `rich` | ≥ 13.9 | Pretty terminal output |
| `pytest` | ≥ 8.0 | Testing framework |

All listed alternatives are **free and open-source**. No paid API is required.
