# ClaimAuditor

A multi-agent LLM system that audits any text passage for factual accuracy. It decomposes text into atomic claims, runs each claim through an adversarial **advocate / skeptic debate** powered by web search, and produces a structured verdict with confidence scores and a narrative summary.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) (orchestration), [FastAPI](https://fastapi.tiangolo.com/) (API), and [Streamlit](https://streamlit.io/) (UI).

---

## How it works

```
Input text
    │
    ▼
Decomposer          — splits text into atomic Claim objects
    │
    ├──────────────────────┐
    ▼                      ▼
Advocate               Skeptic         — run in parallel (LangGraph fan-out)
(web search,           (web search,      operator.add reducers merge their
 supporting)            contradicting)   writes to evidence + debate_transcript
    │                      │
    └──────────┬───────────┘
               ▼
          Arbitrator       — weighs both sides, sets ClaimVerdict + confidence
               │
         ┌─────┴──────────────────────────────┐
         │                                    │
    confidence < 0.4                    confidence ≥ 0.4
    OR both sides > 0.7                 AND rounds < max
    contested                                 │
         │                             more claims? ──► advocate (next claim)
         │                             no more?    ──► reporter
         ▼
      Reporter             — narrative summary, aggregate verdict
```

Each claim gets an independent advocate/skeptic debate loop. The arbitrator can demand additional rounds if confidence is low or both sides score above the conflict threshold. A circuit breaker (`max_rounds`, default 2) prevents infinite loops.

### Agents and models

| Agent       | Model                       | Role                                         |
|-------------|-----------------------------|----------------------------------------------|
| Decomposer  | `claude-haiku-4-5-20251001` | Splits input into atomic claims              |
| Advocate    | `claude-haiku-4-5-20251001` | Web-searches for supporting evidence         |
| Skeptic     | `claude-haiku-4-5-20251001` | Web-searches for contradicting evidence      |
| Arbitrator  | `claude-sonnet-4-6`         | Weighs both sides, issues verdict            |
| Reporter    | `claude-haiku-4-5-20251001` | Aggregates verdicts into narrative summary   |

The Arbitrator uses the stronger Sonnet model — it's the most consequential reasoning step.

### Rulings

Each claim is resolved to one of four rulings:

| Ruling      | Meaning                                      |
|-------------|----------------------------------------------|
| `supported` | Evidence supports the claim                  |
| `refuted`   | Evidence contradicts the claim               |
| `uncertain` | Insufficient evidence to decide              |
| `escalated` | Confidence too low — flagged for human review |

---

## API

The FastAPI server runs at `http://localhost:8000`. Interactive docs at `/docs`.

### `POST /audit`

```json
// Request
{
  "text": "The Eiffel Tower is 330 meters tall and was built in 1887.",
  "source_label": "GPT-4o response",   // optional
  "max_rounds": 2                       // 1–5, default 2
}
```

```json
// Response
{
  "aggregate_verdict": "refuted",
  "aggregate_confidence": 0.82,
  "audit_summary": "One claim is accurate; the construction date is wrong.",
  "escalate_to_human": false,
  "total_claims": 2,
  "source_label": "GPT-4o response",
  "claim_verdicts": [
    {
      "claim_id": "c1",
      "claim_text": "The Eiffel Tower is 330 meters tall.",
      "ruling": "supported",
      "confidence": 0.91,
      "reasoning": "Multiple sources confirm 330 m including antenna.",
      "debate_rounds": 1
    },
    {
      "claim_id": "c2",
      "claim_text": "The Eiffel Tower was built in 1887.",
      "ruling": "refuted",
      "confidence": 0.95,
      "reasoning": "Construction began in 1887 but it was completed in 1889.",
      "debate_rounds": 1
    }
  ]
}
```

### `GET /health`

Returns `{"status": "ok"}`.

---

## Project layout

```
claim-auditor/
  config.py              — model names, thresholds, API keys
  environment.yml        — conda environment
  .env.example           — API key template
  agents/
    decomposer.py        — claim extraction, JSON output
    advocate.py          — web search for supporting evidence
    skeptic.py           — web search for contradicting evidence
    arbitrator.py        — verdict + confidence, JSON output
    reporter.py          — aggregate verdict + narrative summary
  graph/
    state.py             — AuditState TypedDict + sub-types + initial_state()
    graph.py             — StateGraph wiring, conditional edges, compiled graph
  tools/
    search.py            — Tavily wrapper + result formatter
  api/
    schemas.py           — Pydantic request/response models
    main.py              — FastAPI app
  ui/
    app.py               — Streamlit frontend
```

---

## Setup and run

### Step 1 — Create the conda environment (one-time, takes ~1-2 min)

```bash
cd /Users/noah/Github_repos/claim_auditor
conda env create -f environment.yml
```

### Step 2 — Create your .env file

```bash
cp .env.example .env
```

Then open `.env` and fill in your two API keys:

- `ANTHROPIC_API_KEY` — from [console.anthropic.com](https://console.anthropic.com)
- `TAVILY_API_KEY` — from [app.tavily.com](https://app.tavily.com) (free tier works)

### Step 3 — Activate the env

```bash
conda activate claim-auditor
```

### Step 4 — Terminal 1: start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### Step 5 — Terminal 2: start the UI
```bash
conda activate claim-auditor
streamlit run ui/app.py
```

The UI will open at http://localhost:8501 automatically.

---

## Dependencies

- [`langgraph`](https://github.com/langchain-ai/langgraph) — graph orchestration and state management
- [`langchain-anthropic`](https://python.langchain.com/docs/integrations/chat/anthropic/) — LLM calls
- [`tavily-python`](https://docs.tavily.com/) — web search
- [`fastapi`](https://fastapi.tiangolo.com/) + `uvicorn` — API server
- [`streamlit`](https://streamlit.io/) — UI
- [`pydantic`](https://docs.pydantic.dev/) — request/response schemas
