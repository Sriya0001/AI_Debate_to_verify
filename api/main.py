from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from api.schemas import AuditRequest, AuditResponse, ClaimVerdictResponse, SessionSummary
from runner import run_audit
from database.repository import repository

app = FastAPI(
    title="VERITAS AI — Autonomous Multi-Agent Claim Verification",
    description="Multi-agent debate, source credibility scoring, and DeGroot consensus system for factual claim verification.",
    version="2.0.0",
)

# CORS middleware for Streamlit and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "system": "VERITAS AI",
        "version": "2.0.0",
        "description": "Autonomous Multi-Agent Fact Verification Platform",
    }


@app.post("/audit", response_model=AuditResponse)
def execute_audit(request: AuditRequest) -> AuditResponse:
    """
    Executes a multi-agent verification audit on the submitted text.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Audit text cannot be empty.")

    try:
        result = run_audit(
            text=request.text.strip(),
            source_label=request.source_label,
            mode=request.mode,
            max_rounds=request.max_rounds,
            max_research_rounds=request.max_research_rounds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification execution failed: {str(e)}")

    claim_responses = [
        ClaimVerdictResponse(
            claim_id=v.claim_id,
            claim_text=v.claim_text,
            verdict=v.verdict,
            confidence=v.confidence,
            reasoning=v.reasoning,
            key_evidence=v.key_evidence,
            contradicting_evidence=v.contradicting_evidence,
            why_not_other_verdicts=v.why_not_other_verdicts,
            debate_rounds=v.debate_rounds,
            consensus_iterations=v.consensus_iterations,
        )
        for v in result.verdicts
    ]

    return AuditResponse(
        session_id=result.session_id,
        mode=result.mode,
        aggregate_verdict=result.aggregate_verdict,
        aggregate_confidence=result.aggregate_confidence,
        summary=result.summary,
        claims=claim_responses,
        audit_trail=result.audit_trail,
        execution_time_seconds=result.execution_time_seconds,
        total_searches=result.total_searches,
        total_llm_calls=result.total_llm_calls,
    )


@app.get("/audits", response_model=List[SessionSummary])
def list_audits(limit: int = Query(20, ge=1, le=100)) -> List[SessionSummary]:
    records = repository.list_sessions(limit=limit)
    return [
        SessionSummary(
            session_id=r["session_id"],
            input_text=r["input_text"],
            aggregate_verdict=r["aggregate_verdict"],
            aggregate_confidence=r["aggregate_confidence"],
            created_at=r["created_at"],
        )
        for r in records
    ]


@app.get("/audit/{session_id}")
def get_audit(session_id: str) -> Dict[str, Any]:
    session = repository.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")
    return session


@app.get("/audit/{session_id}/evidence")
def get_audit_evidence(session_id: str) -> Dict[str, Any]:
    session = repository.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")
    return {
        "session_id": session_id,
        "evidence": session.get("evidence", []),
    }


@app.get("/audit/{session_id}/debate")
def get_audit_debate(session_id: str) -> Dict[str, Any]:
    session = repository.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")
    return {
        "session_id": session_id,
        "claims": session.get("claims", []),
        "summary": session.get("summary", ""),
    }


@app.get("/audit/{session_id}/consensus")
def get_audit_consensus(session_id: str) -> Dict[str, Any]:
    session = repository.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Audit session not found.")
    return {
        "session_id": session_id,
        "audit_trail": session.get("audit_trail", []),
        "claims": session.get("claims", []),
    }
