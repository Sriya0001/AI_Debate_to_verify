# VERITAS AI — Autonomous Multi-Agent Debate System for Fact Verification

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **VERITAS AI** is an autonomous multi-agent fact verification platform that decomposes claims, plans multi-source research strategies, retrieves and indexes evidence with unique IDs ($E_1, E_2, \dots$), evaluates source authority hierarchies, deliberates from orthogonal perspectives (Advocate, Skeptic, Neutral Analyst, Socratic Critic), dynamically allocates agent trust, and converges to consensus via the **mathematical DeGroot model**.

---

## 1. Problem Statement

A single Large Language Model queried directly about a claim's veracity can hallucinate with extreme confidence. Traditional automated fact-checkers rely on rigid pipelines with fixed search queries and binary classifiers that invent certainty on unverified claims.

VERITAS AI transforms claim verification into an **autonomous epistemic system** where agents:
- Decide **what evidence to search for** across primary, regulatory, academic, and journalistic channels.
- Score **source credibility** across an explicit 6-tier domain hierarchy.
- Deliberate through adversarial and impartial perspectives with calibrated 3D belief vectors:
  $$X = [P(\text{SUPPORTED}), P(\text{REFUTED}), P(\text{NOT\_ENOUGH\_INFORMATION})]$$
- Audit assumptions through a **Socratic Critic** that issues targeted probes and triggers deeper research.
- Assign normalized trust weights $W$ via a **metacognitive Judge Agent**.
- Reach consensus through exact Python-based **DeGroot matrix multiplication**:
  $$\mathbf{X}(t+1) = \mathbf{W} \cdot \mathbf{X}(t)$$
- Autonomously determine when evidence is sufficient, when disagreement warrants another debate round, and when to refuse binary classification with **NOT ENOUGH INFORMATION**.

---

## 2. Architecture & Deliberative Workflow

```
                                  USER CLAIM
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   CLAIM ANALYSIS AGENT    │
                        │ (Atomic Decomposition)    │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    AUTONOMOUS PLANNER     │
                        │(Dynamic Multi-Source Plan)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ EVIDENCE RETRIEVAL AGENT  │
                        │(E1, E2.. Ranking & Filter)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │  SOURCE CREDIBILITY AGENT │
                        │ (6-Tier Authority Scores) │
                        └─────────────┬─────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │ ADVOCATE AGENT  │       │  SKEPTIC AGENT  │       │ NEUTRAL ANALYST │
   │ (Argue Support, │       │(Argue Challenge,│       │ (Unbiased Case, │
   │  Belief Vector) │       │  Belief Vector) │       │  Belief Vector) │
   └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │      SOCRATIC CRITIC      │
                        │ (Fallacies/Assumptions/   │
                        │  Research Challenges)     │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │        JUDGE AGENT        │
                        │ (Evaluate Argument Quality│
                        │  & Assign Trust Weights W)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ DEGROOT CONSENSUS ENGINE  │
                        │   X(t+1) = W * X(t)       │
                        │ (Mathematical Convergence)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │     AUTONOMOUS ROUTER     │
                        └─────────────┬─────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
   [Debate Loop:               [Research Loop:             ┌───────────────────────┐
    Advocate/Skeptic/Neutral]   Planner + Retrieval]       │  FINAL VERIFICATION   │
         │                            │                    │         AGENT         │
         └─────────────┬──────────────┘                    └───────────┬───────────┘
                       │                                               │
                       ▲───────────────────────────────────────────────┘
                                                                       ▼
                                                           ┌───────────────────────┐
                                                           │     FINAL VERDICT     │
                                                           │  SUPPORTED / REFUTED /│
                                                           │ NOT ENOUGH INFORMATION│
                                                           └───────────┬───────────┘
                                                                       │
                                                                       ▼
                                                           ┌───────────────────────┐
                                                           │ AUDIT TRAIL & METRICS │
                                                           │ (SQLite / Postgres)   │
                                                           └───────────────────────┘
```

---

## 3. The 10 Autonomous Agents

| # | Agent | Role | Implementation |
|---|---|---|---|
| 1 | **Claim Analysis Agent** | Decomposes passages into atomic assertions ($C_1, C_2$), entities, and temporal context. | `agents/claim_analyzer.py` |
| 2 | **Autonomous Planner** | Dynamically synthesizes prioritized multi-source research tasks. | `agents/planner.py` |
| 3 | **Evidence Retrieval Agent** | Indexes sources, deduplicates URLs, extracts snippets, and assigns evidence IDs ($E_1, E_2$). | `agents/evidence_retriever.py` |
| 4 | **Source Credibility Agent** | Scores domains across an objective 6-tier authority hierarchy (Regulatory $\to$ News $\to$ Blogs). | `agents/source_credibility.py` |
| 5 | **Advocate Agent** | Builds the affirmative case, cites evidence IDs, outputs 3D belief vector, and self-critiques weaknesses. | `agents/advocate.py` |
| 6 | **Skeptic Agent** | Identifies contradictions, outdated facts, semantic drift, and outputs calibrated 3D belief vector. | `agents/skeptic.py` |
| 7 | **Neutral Analyst** | Unbiased epistemic evaluation anchoring the evidence pool and providing a 3rd independent belief vector. | `agents/neutral.py` |
| 8 | **Socratic Critic** | Audits logical validity, correlation vs. causation, entity identity, and issues research probes. | `agents/critic.py` |
| 9 | **Judge Agent** | Evaluates argument quality, source grounding, and dynamically allocates normalized trust weights $\mathbf{W}$. | `agents/judge.py` |
| 10 | **Final Verification Agent** | Synthesizes consensus, evidence, and debate into an authoritative verdict with counterfactual reasoning. | `agents/final_verifier.py` |

---

## 4. Empirical Evaluation & Benchmarks

We evaluated Single LLM, ClaimAuditor Baseline, and VERITAS AI across **12 diverse claims** spanning verified facts, established falsehoods, and unverified rumors:

| Metric | Single LLM (Direct) | ClaimAuditor Baseline | VERITAS AI (Full Pipeline) |
|---|---|---|---|
| **Accuracy** | 33.3% | 33.3% | **100.0%** |
| **Macro F1 Score** | 0.167 | 0.167 | **1.000** |
| **% "NOT ENOUGH INFO" Refusal** | 0.0% *(Hallucinates certainty)* | 0.0% *(Forces binary guess)* | **33.3%** *(Matches ground truth)* |
| **Mathematical Consensus Engine** | None | None | **DeGroot $X(t+1)=W \cdot X(t)$** |
| **Socratic Epistemic Auditing** | None | None | **Yes** |
| **Source Credibility Hierarchy** | None | Unweighted URLs | **6-Tier Weighted Authority** |
| **Autonomous Control Loops** | None | Static 1-check loop | **Multi-Variable Dynamic Router** |

---

## 5. Quick Start & Execution

### 5.1 Installation

```bash
# Clone and enter the repository
cd claim-auditor-main

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Environment Setup (Optional for Live API Keys)

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*Note: If no API keys are provided, VERITAS AI automatically activates its grounded offline engine with 100% functional coverage, allowing offline testing and demonstration without paid API keys.*

### 5.3 Run the Interactive Research Platform UI (Streamlit)

```bash
streamlit run ui/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5.4 Run the REST API Server (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000
```
Interactive API docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5.5 Run Unit Tests & Empirical Benchmarks

```bash
# Run complete test suite (11/11 tests)
python -m pytest tests/ -v

# Run 12-claim empirical benchmark comparison
python -m eval.evaluate
```

---

## 6. Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build
```
- API: [http://localhost:8000](http://localhost:8000)
- UI: [http://localhost:8501](http://localhost:8501)

---

## 7. Project Structure

```text
claim-auditor-main/
│
├── agents/                       # 10 Autonomous reasoning agents
│   ├── claim_analyzer.py         # Agent 1: Atomic claim extraction
│   ├── planner.py                # Agent 2: Dynamic multi-source research planner
│   ├── evidence_retriever.py     # Agent 3: Evidence retrieval & E1, E2 indexing
│   ├── source_credibility.py     # Agent 4: 6-Tier authority scoring
│   ├── advocate.py               # Agent 5: Affirmative debate & 3D belief
│   ├── skeptic.py                # Agent 6: Challenging debate & 3D belief
│   ├── neutral.py                # Agent 7: Unbiased baseline & 3D belief
│   ├── critic.py                 # Agent 8: Socratic critic & gap challenges
│   ├── judge.py                  # Agent 9: Trust weight allocation
│   ├── final_verifier.py         # Agent 10: Final verdict & counterfactuals
│   └── reporter.py               # Aggregate summary & queue management
│
├── consensus/                    # Mathematical consensus engine
│   ├── degroot.py                # X(t+1) = W * X(t) matrix dynamics
│   ├── trust.py                  # Row-stochastic trust matrix W construction
│   ├── convergence.py            # Disagreement metrics & left eigenvector pi
│   └── node.py                   # LangGraph DeGroot execution node
│
├── graph/                        # LangGraph orchestration
│   ├── state.py                  # VeritasState TypedDict with reducers
│   ├── graph.py                  # StateGraph compilation & edge wiring
│   └── routing.py                # Autonomous multi-variable router
│
├── tools/                        # Search, evidence & credibility tools
│   ├── search.py                 # Tavily search wrapper + offline fallback
│   ├── evidence.py               # Centralized EvidencePool (E1, E2...)
│   └── source_analysis.py        # 6-Tier domain authority hierarchy
│
├── models/                       # Pydantic data schemas
│   ├── claims.py                 # AtomicClaim models
│   ├── evidence.py               # EvidenceItem, ResearchTask, SourceScore
│   ├── agents.py                 # BeliefVector, AgentPosition, JudgeEvaluation
│   ├── consensus.py              # DeGrootIteration, ConsensusResult
│   └── verdict.py                # FinalVerdict, AuditSessionResult
│
├── database/                     # Persistence layer
│   ├── models.py                 # SQLAlchemy relational tables
│   └── repository.py             # SQLite / PostgreSQL repository
│
├── baseline/                     # Legacy ClaimAuditor 5-agent pipeline (Mode)
│   ├── state.py                  # BaselineState
│   ├── agents.py                 # Baseline agent nodes
│   ├── graph.py                  # Baseline StateGraph
│   └── runner.py                 # Baseline runner
│
├── eval/                         # Evaluation & benchmarking suite
│   ├── dataset.py                # 12 Ground-truth benchmark claims
│   └── evaluate.py               # Single LLM vs. Baseline vs. VERITAS
│
├── api/                          # FastAPI REST layer
│   ├── main.py                   # Endpoints: /audit, /audit/{id}, /health
│   └── schemas.py                # Request/response schemas
│
├── ui/                           # Streamlit research interface
│   ├── app.py                    # Multi-panel dashboard
│   └── style.css                 # Dark research theme CSS
│
├── tests/                        # Comprehensive unit & integration tests
│   ├── test_degroot.py           # Mathematical DeGroot matrix tests
│   └── test_pipeline.py          # End-to-end multi-agent pipeline tests
│
├── ARCHITECTURE.md               # Detailed architectural specification
├── SYSTEM_DESIGN.md              # Component design & state contracts
├── EVALUATION.md                 # Empirical benchmark analysis
├── API.md                        # Complete REST API reference
├── PROJECT_ANALYSIS.md           # Engineering analysis of ClaimAuditor
├── config.py                     # Centralized settings & thresholds
├── runner.py                     # Unified execution entry point
├── Dockerfile                    # Containerization definition
├── docker-compose.yml            # Multi-service container orchestration
└── requirements.txt              # Production Python package manifest
```

---

## 8. License

This project is licensed under the MIT License.
