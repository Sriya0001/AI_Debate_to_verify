# VERITAS AI — Architectural Specification

**Autonomous Multi-Agent Debate System for Claim Verification**

---

## 1. Architectural Philosophy

Traditional Large Language Models frequently suffer from hallucinations, sycophancy, and uncalibrated epistemic confidence. When directly queried regarding a claim's veracity, a single model generates an ungrounded answer influenced by training priors without verifying external facts or auditing counter-arguments.

**VERITAS AI** addresses this foundational challenge through an **Autonomous Multi-Agent System (MAS)** combining:
1. **Dynamic Task Formulation & Epistemic Planning**: Autonomous generation of multi-source verification strategies.
2. **Centralized Evidence Pool & Multi-Tier Credibility Modeling**: Systematic indexing ($E_1, E_2, \dots$), relevance filtering, domain authority weighting, and primary/secondary source discernment.
3. **Multi-Perspective Deliberative Teams**: Orthogonal reasoning agents (Advocate, Skeptic, Neutral Analyst, and Socratic Critic) providing calibrated 3D probability distributions over verdict space.
4. **Metacognitive Trust Allocation**: Dynamic, non-arbitrary trust scoring derived from evidence grounding and argumentation quality.
5. **Exact Mathematical DeGroot Consensus Dynamics**: True matrix multiplication $X(t+1) = W \cdot X(t)$ tracking belief convergence across discrete iterations.
6. **Autonomous Dynamic Control Loop**: Multi-variable routing evaluating evidence sufficiency, agent disagreement entropy, and source contradictions to self-correct and retrieve additional evidence when needed.
7. **First-Class "NOT ENOUGH INFORMATION"**: Preventing forced binary classification and refusal to fabricate certainty on unverified claims.

---

## 2. Global System Workflow

```
                                USER CLAIM OR PASSAGE
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │  CLAIM ANALYSIS AGENT   │
                             │ (Atomic Decomposition)  │
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │   AUTONOMOUS PLANNER    │
                             │ (Multi-Source Strategy) │
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │   EVIDENCE RETRIEVAL    │
                             │(E1, E2.. Ranking/Filter)│
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │SOURCE CREDIBILITY AGENT │
                             │ (6-Tier Domain Weights) │
                             └────────────┬────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
         ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
         │ ADVOCATE AGENT  │     │  SKEPTIC AGENT  │     │ NEUTRAL ANALYST │
         │ (Affirmative)   │     │ (Challenging)   │     │  (Impartial)    │
         └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                  │                       │                       │
                  └───────────────────────┼───────────────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │     SOCRATIC CRITIC     │
                             │(Fallacy/Gap Auditing)   │
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │       JUDGE AGENT       │
                             │ (Assign Trust Weights W)│
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │DEGROOT CONSENSUS ENGINE │
                             │   X(t+1) = W * X(t)     │
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │    AUTONOMOUS ROUTER    │
                             └────────────┬────────────┘
                                          │
                 ┌────────────────────────┼────────────────────────┐
                 │                        │                        │
                 ▼                        ▼                        ▼
        [Insufficient Evidence   [High Disagreement       [Consensus Reached
          / Source Conflict]        / Open Probe]           / Max Rounds Met]
                 │                        │                        │
                 ▼                        ▼                        ▼
        [Autonomous Research]      [Debate Loop]         ┌───────────────────┐
                 │                        │              │ FINAL VERIFICATION│
                 └───────────┬────────────┘              │       AGENT       │
                             │                           └─────────┬─────────┘
                             ▲─────────────────────────────────────┘
                                                                   │
                                                                   ▼
                                                         ┌───────────────────┐
                                                         │   FINAL VERDICT   │
                                                         │ (SUPPORTED /      │
                                                         │  REFUTED / NEI)   │
                                                         └─────────┬─────────┘
                                                                   │
                                                                   ▼
                                                         ┌───────────────────┐
                                                         │ AUDIT TRAIL & DB  │
                                                         └───────────────────┘
```

---

## 3. Mathematical Formulation of DeGroot Consensus

Let the reasoning team consist of $N = 4$ agents:
$$\mathcal{A} = \{\text{Advocate}, \text{Skeptic}, \text{Neutral}, \text{Critic}\}$$

Each agent $i \in \mathcal{A}$ maintains a belief vector over the 3 verdict states:
$$\Theta = \{\text{SUPPORTED}, \text{REFUTED}, \text{NOT\_ENOUGH\_INFORMATION}\}$$

$$X_i(t) = \begin{bmatrix} P_i(\text{SUPPORTED}) \\ P_i(\text{REFUTED}) \\ P_i(\text{NOT\_ENOUGH\_INFO}) \end{bmatrix}, \quad \sum_{c=1}^3 X_{ic}(t) = 1, \quad X_{ic}(t) \ge 0$$

The collective belief matrix at iteration $t$ is:
$$\mathbf{X}(t) \in \mathbb{R}^{4 \times 3}$$

### 3.1 Trust Matrix Construction
The Judge computes a normalized trust vector $\mathbf{w} = [w_{\text{adv}}, w_{\text{skep}}, w_{\text{neut}}, w_{\text{crit}}]^T$ where $\sum_k w_k = 1.0$.

Each agent $i$ updates its belief by balancing its own self-confidence $\alpha_i \in (0, 1)$ with external consensus:
$$W_{ij} = (1 - \alpha_i) \cdot \delta_{ij} + \alpha_i \cdot w_j$$
where $\sum_{j=1}^4 W_{ij} = 1.0$ (row-stochastic).

### 3.2 Iterative Consensus Dynamics
At each discrete step $t$:
$$\mathbf{X}(t+1) = \mathbf{W} \cdot \mathbf{X}(t)$$

Convergence is declared when the maximum element-wise variation across all agents drops below threshold $\epsilon = 10^{-4}$:
$$\max_{i, j} \|\mathbf{X}_i(t) - \mathbf{X}_j(t)\|_\infty < \epsilon$$
or upon reaching $t_{\max} = 50$ iterations.

The asymptotic stationary consensus vector is given by the left eigenvector $\boldsymbol{\pi}$ of $\mathbf{W}$ satisfying $\boldsymbol{\pi}\mathbf{W} = \boldsymbol{\pi}$:
$$\mathbf{x}^* = \boldsymbol{\pi} \cdot \mathbf{X}(0)$$

---

## 4. Multi-Tier Source Credibility Hierarchy

| Level | Classification | Example Domains | Base Weight | Rationale |
|---|---|---|---|---|
| **Tier 1** | Official Government / Regulatory Repositories | `.gov`, `.mil`, `sec.gov`, `who.int`, `fda.gov`, `un.org` | $0.95 - 1.00$ | Primary statutory authority with legal accountability. |
| **Tier 2** | Primary Institutional Announcements | `toureiffel.paris`, `cern.ch`, `esa.int`, verified corporate IR | $0.88 - 0.94$ | First-party documentation directly governing the entity. |
| **Tier 3** | Peer-Reviewed Research & Scientific Journals | `nature.com`, `science.org`, `nih.gov`, `arxiv.org`, `.edu` | $0.90 - 0.98$ | Blinded editorial scrutiny, empirical methodology. |
| **Tier 4** | Major Established Independent News Outlets | `reuters.com`, `apnews.com`, `bbc.com`, `wsj.com`, `bloomberg.com` | $0.78 - 0.88$ | Formal journalistic ethics, multi-source verification. |
| **Tier 5** | Curated Reference & Fact-Checking Archives | `britannica.com`, `wikipedia.org`, `snopes.com`, `politifact.com` | $0.65 - 0.75$ | Secondary aggregations with citation trail. |
| **Tier 6** | User-Generated Blogs & Social Forums | `medium.com`, `reddit.com`, `twitter.com`, `quora.com` | $0.15 - 0.40$ | Unvetted assertions requiring primary corroboration. |

---

## 5. Persistence & Observability

- **Database**: SQLite / PostgreSQL relational schema storing:
  - Full audit sessions (`AuditRecord`)
  - Decomposed claims & verdicts (`ClaimRecord`)
  - Centralized evidence items (`EvidenceRecord`)
  - Millisecond-timestamped audit trails (`AuditTrailRecord`)
- **Telemetry**: Real-time tracking of latency, tool search count, LLM calls, and token cost estimates.
