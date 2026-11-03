import uuid
import time
from typing import Dict, Any

from baseline.graph import baseline_graph
from baseline.state import initial_baseline_state
from models.verdict import AuditSessionResult, FinalVerdict, VerdictType
from models.claims import AtomicClaim


def run_baseline_audit(text: str, source_label: str = None, max_rounds: int = 2) -> AuditSessionResult:
    """
    Executes the baseline ClaimAuditor pipeline and returns standardized AuditSessionResult.
    """
    start_time = time.time()
    state = initial_baseline_state(text, source_label)
    state["max_rounds"] = max_rounds

    final_state = baseline_graph.invoke(state)

    session_id = str(uuid.uuid4())[:8]
    ruling_map = {
        "supported": VerdictType.SUPPORTED,
        "refuted": VerdictType.REFUTED,
        "uncertain": VerdictType.NOT_ENOUGH_INFORMATION,
        "escalated": VerdictType.NOT_ENOUGH_INFORMATION,
    }

    verdicts = []
    for cv in final_state.get("claim_verdicts", []):
        cid = cv["claim_id"]
        c_obj = next((c for c in final_state.get("claims", []) if c["id"] == cid), None)
        c_text = c_obj["text"] if c_obj else cid
        v_type = ruling_map.get(cv["ruling"].value, VerdictType.NOT_ENOUGH_INFORMATION)
        verdicts.append(
            FinalVerdict(
                claim_id=cid,
                claim_text=c_text,
                verdict=v_type,
                confidence=cv["confidence"],
                reasoning=cv["reasoning"],
                key_evidence=[],
                contradicting_evidence=[],
                why_not_other_verdicts="Evaluated under Baseline ClaimAuditor heuristic.",
                verification_complete=True,
                debate_rounds=cv["debate_rounds"],
                consensus_iterations=0,
            )
        )

    agg_v = ruling_map.get(
        final_state.get("aggregate_verdict", "uncertain").value,
        VerdictType.NOT_ENOUGH_INFORMATION
    )

    claims = [
        AtomicClaim(
            claim_id=c["id"],
            claim_text=c["text"],
            claim_type="factual",
            entities=[],
            temporal_context=None,
            verification_questions=[],
            source_span=c["source_span"],
        )
        for c in final_state.get("claims", [])
    ]

    exec_time = round(time.time() - start_time, 2)

    return AuditSessionResult(
        session_id=session_id,
        input_text=text,
        mode="baseline",
        claims=claims,
        verdicts=verdicts,
        aggregate_verdict=agg_v,
        aggregate_confidence=final_state.get("aggregate_confidence") or 0.70,
        summary=final_state.get("audit_summary") or "",
        audit_trail=["[Baseline] Decomposer executed", "[Baseline] Advocate & Skeptic debated", "[Baseline] Arbitrator rendered ruling"],
        execution_time_seconds=exec_time,
        total_searches=len(final_state.get("evidence", [])),
        total_llm_calls=5,
        estimated_tokens=1500,
    )
