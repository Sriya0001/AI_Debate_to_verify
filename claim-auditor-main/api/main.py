from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from graph.graph import graph
from graph.state import initial_state, Ruling
from api.schemas import AuditRequest, AuditResponse, ClaimVerdictResponse, ErrorResponse


app = FastAPI(
    title="ClaimAuditor",
    description="Multi-agent LLM fact-checking via structured debate",
    version="0.1.0",
)

# Allow the Streamlit UI to call this API from localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


# Routes

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/audit",
    response_model=AuditResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def run_audit(request: AuditRequest) -> AuditResponse:
    """
    Run a full multi-agent audit on the submitted text.

    The graph runs synchronously here — for production you'd want
    background tasks + polling, but for a demo synchronous is fine
    since the UI will show a spinner.
    """
    state = initial_state(
        input_text=request.text,
        source_label=request.source_label,
    )

    # Override max_rounds from request if provided
    state["max_rounds"] = request.max_rounds

    # Run the graph
    try:
        final_state = graph.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {e}")

    # Check for agent-level errors
    if final_state.get("error"):
        raise HTTPException(status_code=422, detail=final_state["error"])

    if not final_state.get("audit_complete"):
        raise HTTPException(status_code=500, detail="Graph completed without audit_complete flag.")

    # Build claim text lookup
    claim_text_by_id = {c["id"]: c["text"] for c in final_state["claims"]}

    # Assemble response
    verdict_responses = [
        ClaimVerdictResponse(
            claim_id=v["claim_id"],
            claim_text=claim_text_by_id.get(v["claim_id"], ""),
            ruling=v["ruling"],
            confidence=v["confidence"],
            reasoning=v["reasoning"],
            debate_rounds=v["debate_rounds"],
        )
        for v in final_state["claim_verdicts"]
    ]

    return AuditResponse(
        aggregate_verdict=final_state["aggregate_verdict"] or Ruling.UNCERTAIN,
        aggregate_confidence=final_state["aggregate_confidence"] or 0.0,
        audit_summary=final_state["audit_summary"] or "",
        escalate_to_human=final_state["escalate_to_human"],
        claim_verdicts=verdict_responses,
        total_claims=len(final_state["claims"]),
        source_label=final_state["source_label"],
    )