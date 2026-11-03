import uuid
import time
from typing import Optional, Dict, Any

from graph.graph import veritas_graph
from graph.state import initial_veritas_state, VeritasState
from baseline.runner import run_baseline_audit
from models.verdict import AuditSessionResult, FinalVerdict, VerdictType
from database.repository import repository
from tools.evidence import evidence_pool
from utils.metrics import metrics
from utils.audit_trail import audit_trail


def run_audit(
    text: str,
    source_label: Optional[str] = None,
    mode: str = "veritas",
    max_rounds: int = 3,
    max_research_rounds: int = 2,
) -> AuditSessionResult:
    """
    Executes a fact verification session under either VERITAS mode or Baseline mode.

    Args:
        text: Claim or paragraph text to audit.
        source_label: Optional label denoting source (e.g. 'GPT-4o response').
        mode: 'veritas' (10-agent autonomous consensus) or 'baseline' (5-agent ClaimAuditor).
        max_rounds: Max debate rounds.
        max_research_rounds: Max research iteration loops.

    Returns:
        Structured AuditSessionResult with claims, verdicts, DeGroot consensus, metrics, and audit trail.
    """
    metrics.reset()
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]

    # Mode 1: Baseline Mode
    if mode.lower() == "baseline":
        result = run_baseline_audit(text, source_label, max_rounds)
        repository.save_session(result)
        return result

    # Mode 2: VERITAS Mode (Default)
    state = initial_veritas_state(
        input_text=text,
        source_label=source_label,
        pipeline_mode="veritas",
        max_rounds=max_rounds,
        max_research_rounds=max_research_rounds,
    )

    final_state: VeritasState = veritas_graph.invoke(state)

    exec_time = round(time.time() - start_time, 2)
    metrics_summary = metrics.get_summary()

    # Aggregate verdict calculation
    agg_verdict = final_state.get("aggregate_verdict") or VerdictType.NOT_ENOUGH_INFORMATION
    agg_conf = final_state.get("aggregate_confidence") or 0.70
    summary = final_state.get("audit_summary") or "VERITAS deliberation complete."

    session_result = AuditSessionResult(
        session_id=session_id,
        input_text=text,
        mode="veritas",
        claims=final_state.get("atomic_claims", []),
        verdicts=final_state.get("final_verdicts", []),
        aggregate_verdict=agg_verdict,
        aggregate_confidence=agg_conf,
        summary=summary,
        audit_trail=final_state.get("audit_trail", []),
        execution_time_seconds=exec_time,
        total_searches=metrics_summary.get("total_searches", 0),
        total_llm_calls=metrics_summary.get("total_llm_calls", 0),
        estimated_tokens=metrics_summary.get("total_tokens", 0),
    )

    # Persist in database
    try:
        repository.save_session(
            session_result=session_result,
            evidence_items=final_state.get("evidence_pool", []),
        )
    except Exception as e:
        # Non-fatal persistence logging
        pass

    return session_result

if __name__ == "__main__":
    import sys
    sample_text = sys.argv[1] if len(sys.argv) > 1 else "The Eiffel Tower is 330 meters tall and was completed in 1889."
    print(f"\n=======================================================")
    print(f"  VERITAS AI — AUDITING CLAIM")
    print(f"  \"{sample_text}\"")
    print(f"=======================================================\n")
    res = run_audit(sample_text)
    print(f"Verdict:    {res.aggregate_verdict.value if hasattr(res.aggregate_verdict, 'value') else res.aggregate_verdict}")
    print(f"Confidence: {res.aggregate_confidence * 100:.1f}%")
    print(f"Execution:  {res.execution_time_seconds}s")
    print(f"Summary:    {res.summary}")
    print("\nAtomic Claims Audited:")
    for v in res.verdicts:
        print(f" - [{v.claim_id}]: {v.verdict.value} (Confidence: {v.confidence*100:.1f}%) | {v.claim_text}")
    print("\nDeliberation Details:")
    for v in res.verdicts:
        print(f" - [{v.claim_id}]: Debate Rounds: {v.debate_rounds}, DeGroot Iterations: {v.consensus_iterations}")
        print(f"   Supporting Evidence: {v.key_evidence}")
        print(f"   Reasoning: {v.reasoning[:120]}...")
