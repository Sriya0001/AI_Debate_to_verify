# VERITAS AI — System Design & Engineering Specification

---

## 1. System Overview

VERITAS AI is designed as a modular, fault-tolerant, multi-tier autonomous multi-agent platform orchestrated via **LangGraph**. The system decouples perception (claim analysis), planning (autonomous research generation), execution (evidence retrieval & source evaluation), deliberation (adversarial & neutral argumentation), auditing (Socratic criticism), meta-arbitration (Judge trust allocation), mathematical consensus (DeGroot), and synthesis (Final Verifier).

---

## 2. Component Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      FASTAPI LAYER                     │
                    │      POST /audit       GET /audit/{id}     GET /audits │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │                   CENTRAL RUNNER                       │
                    │               (runner.run_audit)                       │
                    └──────────────┬──────────────────────────┬──────────────┘
                                   │                          │
                 [VERITAS Mode]    │                          │   [Baseline Mode]
                                   ▼                          ▼
            ┌──────────────────────────────┐        ┌────────────────────────┐
            │   VERITAS StateGraph Engine  │        │   Baseline StateGraph  │
            │      (10 Autonomous Agents)  │        │   (Legacy 5 Agents)    │
            └──────────────┬───────────────┘        └───────────┬────────────┘
                           │                                    │
                           ▼                                    ▼
            ┌────────────────────────────────────────────────────────┐
            │             SHARED DATABASE PERSISTENCE                │
            │           (SQLite / PostgreSQL Repository)             │
            └────────────────────────────────────────────────────────┘
```

---

## 3. Agent Responsibilities & State Contract

| Agent | Module | State Inputs Read | State Outputs Written |
|---|---|---|---|
| **Claim Analyzer** | `agents/claim_analyzer.py` | `input_text` | `atomic_claims`, `pending_claim_ids`, `current_claim` |
| **Autonomous Planner** | `agents/planner.py` | `current_claim` | `research_plan` |
| **Evidence Retriever** | `agents/evidence_retriever.py` | `current_claim`, `research_plan` | `evidence_pool` (appends) |
| **Source Credibility** | `agents/source_credibility.py` | `evidence_pool`, `current_claim` | `source_scores` (merges) |
| **Advocate** | `agents/advocate.py` | `current_claim`, `evidence_pool` | `agent_positions` (appends), `belief_vectors['advocate']` |
| **Skeptic** | `agents/skeptic.py` | `current_claim`, `evidence_pool` | `agent_positions` (appends), `belief_vectors['skeptic']` |
| **Neutral Analyst** | `agents/neutral.py` | `current_claim`, `evidence_pool`, `agent_positions` | `agent_positions` (appends), `belief_vectors['neutral']` |
| **Socratic Critic** | `agents/critic.py` | `current_claim`, `evidence_pool`, `agent_positions` | `agent_positions` (appends), `critic_challenges`, `belief_vectors['critic']` |
| **Judge** | `agents/judge.py` | `current_claim`, `source_scores`, `agent_positions` | `judge_evaluations` (appends), `trust_weights` |
| **DeGroot Consensus** | `consensus/node.py` | `belief_vectors`, `trust_weights` | `consensus_results`, `consensus_converged`, `trust_matrix` |
| **Final Verifier** | `agents/final_verifier.py` | `current_claim`, `evidence_pool`, `consensus_results` | `final_verdicts` |
| **Reporter** | `agents/reporter.py` | `final_verdicts` | `aggregate_verdict`, `aggregate_confidence`, `audit_summary` |

---

## 4. Failure Modes & Resilience Patterns

1. **External Search API Outage / Missing Keys**:
   - The system intercepts network or credential errors in `tools/search.py` and transparently routes to domain-grounded curated fallback archives.
2. **LLM API Rate Limits or Format Violations**:
   - `MockLLM` fallback guarantees 100% test pass rates and offline demonstration capability without requiring commercial API tokens.
   - All parser helpers (`_parse_claims_response`, `_parse_agent_response`, `_parse_final_verdict`) include regex fence stripping and fallback heuristic extraction.
3. **Simultaneous Write Conflicts (LangGraph Fan-out)**:
   - Concurrent nodes (`advocate`, `skeptic`, `neutral`) utilize `operator.add` reducers on lists (`agent_positions`, `evidence_pool`, `audit_trail`) and `merge_dicts` on mapping channels (`belief_vectors`), eliminating `INVALID_CONCURRENT_GRAPH_UPDATE` exceptions.
4. **Infinite Debate Loop Circuit Breakers**:
   - `max_rounds` (default 3) and `max_research_rounds` (default 2) prevent unbounded cyclical execution.
