# VERITAS AI — Project Analysis & Architecture Specification
**Autonomous Multi-Agent Debate System for Fact Verification**
*Engineering Foundation Analysis: ClaimAuditor Codebase*

---

## 1. Executive Summary

This document presents a comprehensive technical analysis of the attached **ClaimAuditor** repository and details the architectural transformation into **VERITAS AI: Autonomous Multi-Agent Debate System for Claim Verification**.

ClaimAuditor provides a clean starting foundation with LangGraph orchestration, parallel fan-out execution, and an adversarial debate concept. However, its original implementation is limited to a rigid, 2-agent loop (Advocate vs. Skeptic) with a fixed heuristic arbitrator, static web search strings, binary/ternary classification without formal consensus, no autonomous planning, no source credibility modeling, no Socratic critique, and tight coupling to Anthropic API keys.

VERITAS AI transforms this pipeline into a research-grade **Autonomous Multi-Agent System** featuring:
1. **Dynamic Autonomous Planning**: Intelligent decomposition of claims and dynamic generation of multi-source verification tasks.
2. **Centralized Evidence Management & Source Credibility Modeling**: Systematic evidence indexing ($E_1, E_2, \dots$), domain hierarchy evaluation, recency, independence, and bias weighting.
3. **Multi-Perspective Reasoning Team**: Advocate (supporting), Skeptic (challenging), Neutral Analyst (unbiased baseline), and Socratic Critic (epistemic stress-testing).
4. **Metacognitive Judge Agent**: Dynamic trust allocation based on evidence grounding and argumentation quality.
5. **Mathematical DeGroot Consensus Engine**: Exact Python-based matrix consensus ($X(t+1) = W \cdot X(t)$) tracking 3-dimensional belief vectors $[P(\text{SUPPORTED}), P(\text{REFUTED}), P(\text{NOT\_ENOUGH\_INFO})]$ across debate rounds.
6. **Autonomous Dynamic Router**: Non-hardcoded control flow evaluating evidence sufficiency, disagreement entropy, source contradictions, and convergence criteria to decide between further research, additional debate, or final verdict synthesis.
7. **First-Class "Not Enough Information" Verdict**: Preventing hallucinated certainty on unverifiable or under-sourced claims.
8. **Multi-Provider LLM & Persistence Architecture**: Pluggable LLM provider abstraction (Anthropic, OpenAI, local/Groq, and deterministic mock for evaluation) with SQLite/PostgreSQL persistence.
9. **Dual Pipeline Execution**: VERITAS Mode vs. Baseline ClaimAuditor Mode for academic benchmarking and ablation studies.

---

## 2. Existing Architecture Analysis (ClaimAuditor)

### 2.1 Component Overview
The existing ClaimAuditor repository consists of:
```
claim-auditor-main/
  config.py              # Configuration dataclass, pulls from os.environ
  environment.yml        # Conda environment definition
  .env.example           # ANTHROPIC_API_KEY, TAVILY_API_KEY
  agents/
    decomposer.py        # Splits text into atomic claims (Claude Haiku)
    advocate.py          # Searches Tavily for supporting evidence, argues (Claude Haiku)
    skeptic.py           # Searches Tavily for counterarguments, argues (Claude Haiku)
    arbitrator.py        # Evaluates debate, renders verdict (Claude Sonnet)
    reporter.py          # Formats overall summary report (Claude Haiku)
  graph/
    state.py             # AuditState TypedDict with operator.add reducers
    graph.py             # LangGraph StateGraph wiring & routing functions
  tools/
    search.py            # Tavily client wrapper
  api/
    schemas.py           # Pydantic request/response schemas
    main.py              # FastAPI server (POST /audit, GET /health)
  ui/
    app.py               # Streamlit web interface
    style.css            # Dark theme CSS
```

### 2.2 Existing LangGraph Workflow
```
               [START]
                  │
                  ▼
            [decomposer]
                  │
         ┌────────┴────────┐
         ▼                 ▼
    [advocate]         [skeptic]        <-- Parallel fan-out
         │                 │
         └────────┬────────┘            <-- Fan-in join
                  ▼
            [arbitrator]
                  │
                  ▼ (Conditional Edge: route_after_arbitrator)
         ┌────────┴────────────────────┐
         │ (confidence < 0.5           │ (confidence >= 0.5
         │  and rounds < max)          │  or rounds >= max)
         ▼                             ▼
   [advocate, skeptic]          [check_pending]
   (loop same claim)                   │
                                       ▼ (Conditional Edge: route_after_pending_check)
                              ┌────────┴──────────────┐
                              │ (pending claims left) │ (queue empty)
                              ▼                       ▼
                        [advocate, skeptic]      [reporter]
                        (loop next claim)             │
                                                      ▼
                                                    [END]
```

### 2.3 Existing State Definition (`graph/state.py`)
```python
class AuditState(TypedDict):
    input_text: str
    source_label: str | None
    claims: list[Claim]
    pending_claim_ids: list[str]
    evidence: Annotated[list[EvidenceItem], operator.add]
    debate_transcript: Annotated[list[DebateTurn], operator.add]
    current_round: int
    max_rounds: int
    claim_verdicts: list[ClaimVerdict]
    aggregate_verdict: Ruling | None
    aggregate_confidence: float | None
    audit_summary: str | None
    escalate_to_human: bool
    audit_complete: bool
    error: Annotated[str | None, ...]
```

### 2.4 Limitations of the Existing Implementation
1. **No Autonomous Planning**: Search queries are hardcoded as `"evidence supporting: <claim>"` and `"evidence against or counterargument: <claim>"`. The system has no perception of entity types, temporal context, missing regulatory filings, or primary vs. secondary sources.
2. **Lack of Evidence Identity and Centralization**: Evidence items are loosely collected snippets without unique IDs ($E_1, E_2$), without relevance scoring, and without source domain credibility analysis.
3. **Binary Adversarial Polarization**: Only Advocate and Skeptic debate; there is no unbiased Neutral Analyst to anchor evidence, and no Socratic Critic to question methodological fallacies or semantic drift.
4. **Arbitrary Verdict Synthesis without Consensus**: The Arbitrator acts as a monolithic LLM prompt that guesses confidence and ruling. There is no mathematical consensus model, no trust calibration, and no iterative belief tracking.
5. **Rigid Control Flow**: Loop termination relies on a single `confidence < 0.5` check. It cannot distinguish between "sources are conflicting", "more evidence needed", "high agent disagreement", or "unverifiable claim".
6. **No First-Class 'NOT ENOUGH INFORMATION'**: Claims with zero evidence often get forced into `uncertain` or an uncalibrated low-confidence guess.
7. **Hardcoded Vendor Dependency**: `config.py` crashes on import if `ANTHROPIC_API_KEY` is absent. There is no abstraction for OpenAI, local models, or deterministic offline testing.
8. **No Audit Trail or Live Observability**: Streamlit blocks synchronously until completion. The user cannot see live agent deliberations, belief updates, or consensus convergence.

---

## 3. VERITAS AI Target Architecture

```
                                  USER INPUT
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   CLAIM ANALYSIS AGENT    │
                        │ (Entity/Temporal/Questions)│
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
                        │(Domain/Authority/Recency) │
                        └─────────────┬─────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │    ADVOCATE     │       │     SKEPTIC     │       │ NEUTRAL ANALYST │
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
         │ (High Disagreement /       │ (Insufficient Evidence /   │ (Consensus Reached /
         │  Unresolved Challenges)    │  Source Conflict)          │  Max Rounds Met)
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
                                                           │ AUDIT TRAIL + METRICS │
                                                           │ (SQLite / Postgres)   │
                                                           └───────────────────────┘
```

---

## 4. Detailed Specification of the Required Agents

### Agent 1: Claim Analysis Agent (`agents/claim_analyzer.py`)
- **Objective**: Decompose compound text passages into atomic, independently verifiable claim entities with rich contextual metadata.
- **Input**: Raw text passage.
- **Output Schema**:
  ```python
  class AtomicClaim(BaseModel):
      claim_id: str                      # e.g., "C1", "C2"
      claim_text: str                    # Atomic factual assertion
      claim_type: str                    # "factual", "numerical", "temporal", "causal"
      entities: list[str]                # Key subjects/organizations/places
      temporal_context: str | None       # Year, era, or timeframe
      verification_questions: list[str]  # 2-3 targeted sub-questions
      source_span: str                   # Exact substring from user input
  ```

### Agent 2: Autonomous Planner (`agents/planner.py`)
- **Objective**: Synthesize targeted, multi-angle research strategies rather than blindly searching one generic string.
- **Functions**:
  - Analyzes the claim's entities, temporal context, and domain.
  - Determines required evidence types: primary sources (official statements, SEC filings, government gazettes), peer-reviewed research, major journalistic outlets, or fact-checking archives.
  - Generates parameterized `ResearchTask` objects with explicit rationale and priority.
- **Output Schema**:
  ```python
  class ResearchTask(BaseModel):
      task_id: str
      claim_id: str
      query: str
      target_source_type: str     # "official", "regulatory", "academic", "news"
      purpose: str                # e.g., "Verify acquisition date from regulatory filing"
      priority: int               # 1 (critical) to 3 (supplementary)
  ```

### Agent 3: Evidence Retrieval & Management Agent (`agents/evidence_retriever.py` & `tools/evidence.py`)
- **Objective**: Execute research tasks, deduplicate sources, extract granular snippets, and maintain a centralized indexed evidence pool.
- **Data Model**:
  ```python
  class EvidenceItem(BaseModel):
      evidence_id: str             # e.g., "E1", "E2"
      claim_id: str
      task_id: str
      title: str
      url: str
      snippet: str
      source_domain: str
      retrieved_at: str
      supports_claim: bool | None  # Initial stance tagging
      relevance_score: float       # 0.0 - 1.0
  ```
- **Resilience**: Integrated Tavily search with automatic fallback to mock/cached search when offline or when API credits are exhausted, guaranteeing 100% testability.

### Agent 4: Source Credibility Agent (`agents/source_credibility.py`)
- **Objective**: Compute an objective credibility score for every retrieved source based on an explicit authority hierarchy and verifiable criteria.
- **Authority Hierarchy**:
  1. Official Government / Regulatory / Legal Repositories ($0.95 - 1.0$)
  2. Primary Corporate / Institutional Announcements ($0.85 - 0.92$)
  3. Peer-Reviewed Academic & Scientific Publications ($0.90 - 0.98$)
  4. Major Established News Organizations ($0.75 - 0.88$)
  5. Secondary Publications & Niche Media ($0.55 - 0.70$)
  6. Unverified Blogs, Forums & Low-Authority Domains ($0.15 - 0.40$)
- **Output**: Detailed `SourceScore` containing domain score, authority level, independence evaluation, recency factor, and rationale.

### Agent 5: Advocate Agent (`agents/advocate.py`)
- **Objective**: Construct the rigorous affirmative case defending the claim.
- **Requirements**:
  - Grounds every argument in specific evidence IDs ($E_i$).
  - Outputs calibrated probability distribution:
    $$P = [p_{\text{supp}}, p_{\text{ref}}, p_{\text{unk}}], \quad \sum p_i = 1$$
  - Explicitly states weaknesses and limitations in its own affirmative argument.

### Agent 6: Skeptic Agent (`agents/skeptic.py`)
- **Objective**: Construct the rigorous counter-argument and stress-test the claim.
- **Requirements**:
  - Actively uncovers counter-evidence, missing temporal context, semantic drift, or numerical discrepancies.
  - Cites specific evidence IDs ($E_j$).
  - Outputs calibrated 3D probability distribution.
  - Discloses weaknesses in its counter-argument.

### Agent 7: Neutral Analyst Agent (`agents/neutral.py`)
- **Objective**: Deliver an unbiased, independent perspective grounded in the overall evidence pool.
- **Role**: Prevents bipartisan polarization between Advocate and Skeptic. Acts as an independent third belief vector $[p_{\text{supp}}, p_{\text{ref}}, p_{\text{unk}}]$.

### Agent 8: Socratic Critic Agent (`agents/critic.py`)
- **Objective**: Epistemic and logical audit of the debate.
- **Evaluation Criteria**:
  - Entity match integrity (are agents arguing about the exact same entity?).
  - Temporal consistency (is the evidence fresh or superseded?).
  - Logical fallacies (correlation vs. causation, affirming the consequent, hasty generalization).
  - Identification of missing evidence and generation of `NewResearchRequest` tasks if gaps exist.

### Agent 9: Judge Agent (`agents/judge.py`)
- **Objective**: Metacognitive arbitration and dynamic trust weight assignment.
- **Mechanism**:
  - Evaluates the rigor of each reasoning agent (Advocate, Skeptic, Neutral, Critic).
  - Assesses source quality cited by each agent.
  - Computes a normalized trust vector $\mathbf{w} = [w_{\text{adv}}, w_{\text{skep}}, w_{\text{neut}}, w_{\text{crit}}]$ such that $\sum w_k = 1.0$ and $w_k \ge 0$.
  - Generates explicit justification for each agent's trust allocation.

### Agent 10: Final Verification Agent (`agents/final_verifier.py`)
- **Objective**: Authoritative verdict synthesis.
- **Output**:
  - `verdict`: `SUPPORTED`, `REFUTED`, or `NOT_ENOUGH_INFORMATION`.
  - `confidence`: Calibrated float ($0.0 - 1.0$).
  - `reasoning`: Comprehensive, multi-paragraph synthesis.
  - `key_evidence`: List of decisive supporting evidence IDs.
  - `contradicting_evidence`: List of decisive contradicting evidence IDs.
  - `why_not_other_verdicts`: Counterfactual explanation explaining why the alternative verdicts were rejected.

---

## 5. Mathematical DeGroot Consensus Engine

### 5.1 Formulation
Let $K = 4$ be the set of reasoning agents:
$$\mathcal{A} = \{\text{Advocate}, \text{Skeptic}, \text{Neutral}, \text{Critic}\}$$

Each agent $i \in \mathcal{A}$ holds a belief vector over the 3 verdict states:
$$\Theta = \{\text{SUPPORTED}, \text{REFUTED}, \text{NOT\_ENOUGH\_INFORMATION}\}$$

$$X_i(t) = \begin{bmatrix} P_i(\text{SUPPORTED}) \\ P_i(\text{REFUTED}) \\ P_i(\text{NOT\_ENOUGH\_INFO}) \end{bmatrix}, \quad \text{where } \sum_{c=1}^3 X_{ic}(t) = 1, \quad X_{ic}(t) \ge 0$$

The collective belief matrix at iteration $t$ is:
$$\mathbf{X}(t) \in \mathbb{R}^{K \times 3}$$

### 5.2 Trust Matrix Construction
The Judge computes normalized trust weights $\mathbf{w} = [w_1, w_2, w_3, w_4]^T$.
Each agent $i$ updates its belief by weighting its own prior belief with the opinions of others according to trust matrix $\mathbf{W} \in \mathbb{R}^{K \times K}$:
$$W_{ij} = (1 - \alpha_i) \cdot \delta_{ij} + \alpha_i \cdot w_j$$
where $\alpha_i \in (0, 1)$ is agent $i$'s openness to external consensus, and $\sum_{j=1}^K W_{ij} = 1$ (row-stochastic).

### 5.3 Iterative Dynamics & Convergence
At each discrete step $t$:
$$\mathbf{X}(t+1) = \mathbf{W} \cdot \mathbf{X}(t)$$

Convergence is reached when the maximum variation across all agents drops below the convergence threshold $\epsilon$ (default $10^{-4}$):
$$\max_{i, j} \|\mathbf{X}_i(t) - \mathbf{X}_j(t)\|_\infty < \epsilon$$
or when the iteration limit $t_{\max}$ (default 50) is reached.

The stationary consensus vector is:
$$\mathbf{x}^* = \lim_{t \to \infty} \mathbf{X}_i(t)$$

Every round of belief evolution is recorded in `consensus_history` for real-time visualization.

---

## 6. Autonomous Loop & Dynamic Routing Logic

Unlike ClaimAuditor's static loop, VERITAS AI evaluates multi-dimensional criteria after each debate round:

```python
def route_veritas_debate(state: VeritasState) -> str:
    # 1. Check evidence sufficiency
    if len(state["evidence_pool"]) < config.min_evidence_count:
        if state["research_rounds"] < config.max_research_rounds:
            return "autonomous_research"
        else:
            return "final_verification" # Will lead to NOT_ENOUGH_INFORMATION

    # 2. Check for high agent disagreement (Shannon entropy / variance)
    disagreement = compute_belief_disagreement(state["belief_vectors"])
    
    # 3. Check for conflicting high-credibility sources
    source_conflict = detect_high_credibility_conflict(state["evidence_pool"], state["source_scores"])

    # 4. Check if Socratic Critic raised unresolved research challenges
    unresolved_challenges = len(state.get("critic_research_requests", [])) > 0

    if (source_conflict or unresolved_challenges) and state["research_rounds"] < config.max_research_rounds:
        return "autonomous_research"

    if disagreement > config.conflict_threshold and state["current_round"] < state["max_rounds"]:
        return "debate_loop"

    # 5. Consensus converged or reached max rounds
    return "final_verification"
```

---

## 7. Migration & Engineering Plan

### Phase 1: Foundation & Data Models
- Create clean top-level architecture:
  - `models/`: Pydantic models for claims, evidence, credibility, beliefs, consensus, verdicts.
  - `consensus/`: Mathematical DeGroot algorithm, trust matrix calculations, convergence detectors.
  - `tools/`: Dynamic search, evidence indexing, source domain authority heuristics, offline fallbacks.
  - `utils/`: LLM provider abstraction, audit trail logger, metrics tracker.

### Phase 2: Agent Implementations
- Refactor existing Decomposer into `ClaimAnalyzerAgent`.
- Implement `PlannerAgent`.
- Implement `EvidenceRetrievalAgent`.
- Implement `SourceCredibilityAgent`.
- Refactor `AdvocateAgent` and `SkepticAgent` to support 3D belief distributions, evidence citations, and weakness disclosures.
- Implement `NeutralAnalystAgent`.
- Implement `SocraticCriticAgent`.
- Refactor Arbitrator into `JudgeAgent` with dynamic trust weight computation.
- Implement `FinalVerifierAgent`.

### Phase 3: LangGraph Orchestration & Dynamic Routing
- Overhaul `graph/state.py` into `VeritasState` with complete typed schemas and reducers.
- Implement `graph/graph.py` with parallel execution of reasoning agents, iterative research loops, and DeGroot consensus node.
- Preserve the legacy ClaimAuditor graph as `BaselineGraph` to enable runtime switching and comparison.

### Phase 4: API, Persistence & Real-Time Tracking
- Upgrade `api/main.py` with `/audit`, `/audit/{id}`, `/audit/{id}/evidence`, `/audit/{id}/debate`, `/audit/{id}/consensus`, and `/health`.
- Implement `database/repository.py` with SQLite backing store.
- Include execution time, token estimates, search count, and debate rounds in response metrics.

### Phase 5: Streamlit Research Platform UI
- Enhance `ui/app.py` with:
  - Live agent activity progression.
  - Final verdict dashboard with confidence meter and counterfactual reasoning.
  - Agent belief matrix table.
  - DeGroot consensus convergence chart.
  - Interactive evidence explorer with credibility scores and relevance badges.
  - Formatted debate transcript (Advocate, Skeptic, Neutral, Critic, Judge).
  - Real-time audit trail logs.
  - Pipeline Mode toggle: **VERITAS Mode** vs. **Baseline Mode**.

### Phase 6: Verification, Testing & Academic Evaluation
- Implement unit tests for:
  - Atomic claim decomposition.
  - Evidence schema validation.
  - Probability simplex constraints ($\sum p_i = 1$).
  - Row-stochastic trust matrix validation ($\sum_j W_{ij} = 1$).
  - Mathematical DeGroot matrix multiplication and convergence.
  - Autonomous dynamic routing logic.
  - Final verifier schema integrity.
- Build benchmarking script comparing Single LLM vs. Baseline vs. VERITAS AI across accuracy, calibration, debate rounds, and cost.

---

## 8. Verification Matrix & Expected Outcomes

| Example Claim | Expected Verdict | Autonomous Behavior Observed |
|---|---|---|
| *"The Eiffel Tower is 330 meters tall and was built in 1889."* | `SUPPORTED` | Planner retrieves official documentation; consensus converges rapidly; high confidence (>85%). |
| *"The Moon is made of green cheese and was colonized in 1969."* | `REFUTED` | Skeptic retrieves scientific refutation; Critic flags physical impossibility; consensus converges to Refuted (>90%). |
| *"Project Titan discovered sentient extraterrestrial signals in deep space in March 2026."* | `NOT ENOUGH INFORMATION` | Search yields no authoritative primary or scientific sources; Credibility agent flags lack of verified evidence; Final Verifier refuses to hallucinate certainty. |
| *Multi-claim paragraph containing mixed facts* | Independent verdicts per claim | Claim Analyzer splits paragraph into $C_1, C_2 \dots$; each claim undergoes independent debate and DeGroot consensus. |

---
*VERITAS AI — Advanced Autonomous Multi-Agent Claim Verification System*
