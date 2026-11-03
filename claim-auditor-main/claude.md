# ClaimAuditor — Project Context for Claude Code

## What this project is

A multi-agent LLM system that audits any text passage for factual accuracy.
It decomposes text into atomic claims, runs each claim through an adversarial
advocate/skeptic debate powered by web search, and produces a structured verdict
with confidence scores and a narrative summary.

Built with LangGraph (orchestration), FastAPI (API layer), and Streamlit (UI).

## Why it was built

Side project to explore the LLM-as-judge / multi-agent debate space.
Intentionally sits at the intersection of traditional ML evaluation thinking
and modern agent architecture. Designed to be presentable and demonstrable —
not just a research artifact.

---

## Architecture overview

```
Input text
    │
    ▼
Decomposer          — splits text into atomic Claim objects
    │
    ├──────────────────────┐
    ▼                      ▼
Advocate               Skeptic         — run IN PARALLEL (LangGraph fan-out)
(web search,           (web search,      operator.add reducers merge their
 supporting)            contradicting)   writes to evidence + debate_transcript
    │                      │
    └──────────┬───────────┘
               ▼
          Arbitrator       — weighs both sides, sets ClaimVerdict + confidence
               │
         ┌─────┴──────────────────────────────┐
         │                                    │
    confidence < 0.4                    confidence >= 0.4
    OR both sides > 0.7                 AND rounds < max
    contested                                 │
         │                             ┌──────┴──────┐
         │                        more rounds?   no more rounds?
         │                             │               │
         │                        advocate        check_pending
         │                        (loop back)          │
         │                                    more claims? ──► advocate (next claim)
         │                                    no more?    ──► reporter
         ▼
      Reporter             — narrative summary, aggregate verdict
```

## LangGraph key concepts used

- **StateGraph** with `AuditState` (TypedDict) flowing through all nodes
- **`operator.add` reducers** on `evidence` and `debate_transcript` — allows
  advocate and skeptic to run in parallel and append without race conditions
- **Conditional edges** — `route_after_arbitrator` and `route_after_pending_check`
  are plain Python functions that read state and return a routing string
- **Fan-out / fan-in** — `add_edge("decomposer", "advocate")` +
  `add_edge("decomposer", "skeptic")` runs both in parallel;
  `add_edge(["advocate", "skeptic"], "arbitrator")` waits for both
- **Cycles** — the graph loops back to advocate when confidence is low
  or more claims remain; `pending_claim_ids` and `current_round` control termination

## State design (`graph/state.py`)

Key design decisions:
- `pending_claim_ids: list[str]` — drives the per-claim processing loop.
  Populated by decomposer, first item is always the "current" claim,
  popped by `_check_pending_node` after each claim is resolved.
- `current_round: int` — reset to 0 by `_check_pending_node` between claims.
  Incremented by arbitrator after each round.
- `max_rounds: int` — circuit breaker, default 2. Prevents infinite loops.
- `evidence: Annotated[list[EvidenceItem], operator.add]` — parallel-safe append
- `debate_transcript: Annotated[list[DebateTurn], operator.add]` — parallel-safe append
- `claim_verdicts: list[ClaimVerdict]` — NO reducer. Arbitrator manages full list,
  filtering out prior verdicts for the same claim before appending latest.
- `audit_summary: str | None` — narrative output from reporter, added late in build.
  Make sure initial_state() includes `audit_summary=None`.

## Agent responsibilities

| Agent       | Model                    | Reads from state                        | Writes to state                          |
|-------------|--------------------------|------------------------------------------|-------------------------------------------|
| Decomposer  | claude-haiku-4-5-20251001 | input_text                             | claims, pending_claim_ids                |
| Advocate    | claude-haiku-4-5-20251001 | pending_claim_ids, claims, transcript  | evidence (append), debate_transcript (append) |
| Skeptic     | claude-haiku-4-5-20251001 | pending_claim_ids, claims, transcript  | evidence (append), debate_transcript (append) |
| Arbitrator  | claude-sonnet-4-6         | all of above + evidence                | claim_verdicts, current_round, escalate_to_human |
| Reporter    | claude-haiku-4-5-20251001 | claim_verdicts, claims, input_text     | aggregate_verdict, aggregate_confidence, audit_summary, audit_complete |

**Important**: Arbitrator uses `claude-sonnet-4-6` (stronger model) — it's the most consequential
reasoning step. All others use `claude-haiku-4-5-20251001`.

## Prompt patterns

- **Decomposer**: JSON array output, claims split at atomic level.
  IDs assigned in Python (c1, c2...) NOT by the LLM — LLMs produce inconsistent ID formats.

- **Advocate / Skeptic**: `---ARGUMENT---` separator between argument text and
  confidence float. Avoids JSON parsing issues when argument prose contains
  quotes or braces. Confidence is clamped to [0.0, 1.0] in `_parse_response`.

- **Arbitrator**: JSON output `{ruling, confidence, reasoning}`. Uses `claude-sonnet-4-6`
  with `temperature=0` for deterministic verdicts.

- **All agents**: Markdown fence stripping in parse helpers — models ignore
  "no markdown fences" instructions ~10-15% of the time.

## Error handling pattern

Agents return `{"error": "message"}` rather than raising exceptions.
This is the LangGraph pattern — raising inside a node crashes the whole run.
The API layer checks `final_state.get("error")` and raises HTTPException.
Reporter is the exception: its LLM failure is non-fatal since structured
verdicts are already complete. It catches and stuffs the error into the narrative.

## What is NOT yet built

- `tests/` — pytest unit tests for each agent and graph-level evals.
  An `evals/` subdirectory with a small dataset of claims with known
  truth values would make the project demonstrable and publishable.
  Priority: high — this is what makes the project credible to present.

- Config wiring — agents currently hardcode model names in `_get_llm()`.
  They should read from `config.config` instead. One-line change per agent,
  do this before anything else.

- Debate transcript endpoint — `ui/app.py` has an "About this audit" expander
  that notes the transcript isn't exposed yet. Add a `/audit/transcript` endpoint
  (or include `debate_transcript` in the main `/audit` response) and wire it into
  a Streamlit expander showing advocate/skeptic turns per claim. High demo value.

- Tool upgrade to Option C — advocate and skeptic currently do web search only.
  The planned upgrade is "reasoning first, search as a tool" — the LLM decides
  whether to call search based on whether it needs grounding. This requires
  converting advocate/skeptic to use LangChain tool-calling rather than
  hardcoded search calls. Do this after tests are green.

## Running the project

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate claim-auditor

# 2. Set API keys
cp .env.example .env
# edit .env with real ANTHROPIC_API_KEY and TAVILY_API_KEY

# 3. Start API
uvicorn api.main:app --reload --port 8000

# 4. Start UI (separate terminal)
streamlit run ui/app.py
```

## Dependencies

- `langgraph` — graph orchestration
- `langchain-anthropic` — LLM calls
- `langchain-core` — message types (SystemMessage, HumanMessage)
- `tavily-python` — web search
- `fastapi` + `uvicorn` — API server
- `streamlit` — UI
- `pydantic` — request/response schemas

## UI (`ui/app.py`)

Dark editorial aesthetic — Playfair Display wordmark, IBM Plex Mono for
labels/data, IBM Plex Sans for body. Designed to feel like a serious research
tool, not a chatbot demo.

Key UI components:
- Textarea input + "Run Audit" button → POSTs to `/audit` with spinner
- Escalation banner — only renders when `escalate_to_human` is True
- Aggregate block — overall verdict badge, weighted confidence bar, narrative summary
- Ruling count summary (supported/refuted/uncertain/escalated counts)
- Per-claim cards — left border color encodes ruling, confidence bar, arbitrator reasoning
- "About this audit" expander — model info, round count, source label

Color encoding (consistent throughout):
- supported  → green  (#4ade80 / #166534)
- refuted    → red    (#f87171 / #991b1b)
- uncertain  → amber  (#fbbf24 / #92400e)
- escalated  → purple (#a78bfa / #5b21b6)

The debate transcript is NOT yet shown in the UI. The expander has a note
pointing to the missing `/audit/transcript` endpoint as the next step.

Settings sidebar: max_rounds slider (1-5), optional source_label text input.

## Key files

```
claim-auditor/
  CLAUDE.md              — this file; Claude Code context
  config.py              — model names, thresholds, API keys from env
  environment.yml        — conda environment
  .env.example           — API key template
  agents/
    __init__.py
    decomposer.py        — claim extraction, claude-haiku-4-5-20251001, JSON output
    advocate.py          — web search for support, ---ARGUMENT--- separator
    skeptic.py           — web search for contradiction, ---ARGUMENT--- separator
    arbitrator.py        — verdict + confidence, claude-sonnet-4-6, JSON output
    reporter.py          — aggregate verdict, narrative summary
  graph/
    __init__.py
    state.py             — AuditState TypedDict + sub-types + initial_state()
    graph.py             — StateGraph wiring, conditional edges, compiled graph
  tools/
    __init__.py
    search.py            — Tavily wrapper + result formatter
  api/
    __init__.py
    schemas.py           — Pydantic request/response models
    main.py              — FastAPI app, /audit endpoint
  ui/
    app.py               — Streamlit frontend (BUILT)
```

## Decisions worth preserving

1. `TypedDict` not Pydantic for state — LangGraph works with TypedDict natively,
   no validation overhead on every state update.

2. `(str, Enum)` double inheritance for Ruling/EvidenceType — makes enum members
   equal to their string values, so JSON serialization and LLM string comparison
   both work without conversion.

3. `check_pending` is a real node, not an edge function — it mutates state
   (pops pending_claim_ids, resets current_round). State mutations belong in
   nodes; edge functions should be pure reads that return a routing string.

4. Module-level `graph = build_graph()` singleton — validates graph structure
   at import time, not at first request.

5. Arbitrator filters prior verdicts before appending — on loop iterations,
   prevents duplicate verdicts accumulating for the same claim.

6. Weighted confidence by claim text length — crude but reasonable proxy for
   claim complexity. Better than flat averaging.