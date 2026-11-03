# VERITAS AI — REST API Documentation

The VERITAS AI API server is implemented with **FastAPI** and provides high-performance endpoints for claim verification, asynchronous task monitoring, persistent audit inspection, evidence exploration, and consensus telemetry.

- **Base URL**: `http://localhost:8000`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## 1. Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check and status telemetry |
| `POST` | `/audit` | Run autonomous multi-agent verification on input text |
| `GET` | `/audits` | List recent persistent audit sessions |
| `GET` | `/audit/{id}` | Retrieve complete audit session with claims and verdicts |
| `GET` | `/audit/{id}/evidence` | Retrieve centralized evidence pool for a session |
| `GET` | `/audit/{id}/debate` | Retrieve debate transcript and claim breakdown |
| `GET` | `/audit/{id}/consensus` | Retrieve DeGroot consensus metrics and audit trail |

---

## 2. API Endpoints Specification

### 2.1 `GET /health`

**Response:**
```json
{
  "status": "ok",
  "system": "VERITAS AI",
  "version": "2.0.0",
  "description": "Autonomous Multi-Agent Fact Verification Platform"
}
```

---

### 2.2 `POST /audit`

Executes the full 10-agent deliberative verification pipeline.

**Request Schema:**
```json
{
  "text": "The Eiffel Tower is 330 meters tall and was built in 1889.",
  "source_label": "GPT-4o response",
  "mode": "veritas",
  "max_rounds": 3,
  "max_research_rounds": 2
}
```

**Response Schema:**
```json
{
  "session_id": "a1b2c3d4",
  "mode": "veritas",
  "aggregate_verdict": "SUPPORTED",
  "aggregate_confidence": 0.91,
  "summary": "OVERALL VERDICT: SUPPORTED (Confidence: 91%)\n\nCLAIM BREAKDOWN:\n• [SUPPORTED] \"The Eiffel Tower is 330 meters tall\" (Confidence: 91%) — Multiple institutional records confirm official height.\n• [SUPPORTED] \"The Eiffel Tower was built in 1889\" (Confidence: 90%) — Completed and inaugurated for the 1889 World's Fair.",
  "claims": [
    {
      "claim_id": "C1",
      "claim_text": "The Eiffel Tower is 330 meters tall.",
      "verdict": "SUPPORTED",
      "confidence": 0.91,
      "reasoning": "Official architectural records from the operating company confirm 330m height including telecommunications antennas.",
      "key_evidence": ["E1", "E2"],
      "contradicting_evidence": [],
      "why_not_other_verdicts": "Refuted is rejected due to lack of counter-evidence; Not Enough Information is rejected because official primary sources exist.",
      "debate_rounds": 1,
      "consensus_iterations": 4
    }
  ],
  "audit_trail": [
    "[10:30:02] Claim Analyzer: Extracted 2 atomic claim(s): [C1, C2]",
    "[10:30:03] Autonomous Planner: Generated 3 targeted research tasks for claim C1",
    "[10:30:04] Evidence Retriever: Retrieved 4 evidence item(s) [E1 - E4] for claim C1",
    "[10:30:05] Source Credibility Agent: Evaluated 3 source domain(s) (2 high-authority)",
    "[10:30:06] Advocate: P(Supp)=0.85, P(Ref)=0.08, Cited: ['E1', 'E2']",
    "[10:30:06] Skeptic: P(Supp)=0.68, P(Ref)=0.18, Cited: ['E2']",
    "[10:30:07] Neutral Analyst: Independent P(Supp)=0.82, P(Ref)=0.08, P(Unk)=0.10",
    "[10:30:08] Socratic Critic: Issued 1 probe(s) (0 research request(s))",
    "[10:30:09] Judge: Assigned trust weights: [advocate: 0.32, skeptic: 0.28, neutral: 0.25, critic: 0.15]",
    "[10:30:10] DeGroot Consensus Engine: Converged=True in 4 iteration(s)",
    "[10:30:11] Final Verification Agent: Approved verdict for C1: [SUPPORTED] with 91% confidence"
  ],
  "execution_time_seconds": 0.18,
  "total_searches": 4,
  "total_llm_calls": 9
}
```

---

### 2.3 `GET /audit/{session_id}`

Retrieves complete persistent audit data from SQLite/PostgreSQL storage.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/audit/a1b2c3d4"
```

---

### 2.4 `GET /audit/{session_id}/evidence`

Retrieves all indexed evidence items ($E_1, E_2, \dots$) associated with the audit session.

**Response Schema:**
```json
{
  "session_id": "a1b2c3d4",
  "evidence": [
    {
      "evidence_id": "E1",
      "claim_id": "C1",
      "url": "https://www.toureiffel.paris/en/the-monument/key-figures",
      "title": "Key Figures & Dimensions - Official Eiffel Tower Website",
      "snippet": "The Eiffel Tower stands at 330 meters (1,083 ft) tall including antennas. Construction began on 28 January 1887 and was completed on 31 March 1889.",
      "source_domain": "toureiffel.paris",
      "supports_claim": true,
      "relevance_score": 0.98
    }
  ]
}
```

---

### 2.5 `GET /audits`

Lists recent audits with timestamps, summary verdicts, and confidence ratings.
