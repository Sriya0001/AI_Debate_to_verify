from typing import Dict, Any, List
from graph.state import VeritasState
from models.verdict import VerdictType
from utils.audit_trail import format_audit_log


def reporter_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Computes overall aggregate verdict and weighted confidence
    across all claims, producing a clear synthesis report.
    """
    verdicts = state.get("final_verdicts", [])
    if not verdicts:
        return {
            "aggregate_verdict": VerdictType.NOT_ENOUGH_INFORMATION,
            "aggregate_confidence": 0.0,
            "audit_summary": "No claims evaluated.",
            "verification_complete": True,
            "error": "No verdicts generated",
        }

    # Aggregate Verdict Logic:
    # 1. Any REFUTED claim -> Aggregate is REFUTED (a falsehood falsifies the passage)
    # 2. Any NOT_ENOUGH_INFORMATION -> Aggregate is NOT_ENOUGH_INFORMATION
    # 3. All claims SUPPORTED -> Aggregate is SUPPORTED
    ruling_types = [v.verdict for v in verdicts]
    if VerdictType.REFUTED in ruling_types:
        agg_verdict = VerdictType.REFUTED
    elif VerdictType.NOT_ENOUGH_INFORMATION in ruling_types:
        agg_verdict = VerdictType.NOT_ENOUGH_INFORMATION
    elif all(r == VerdictType.SUPPORTED for r in ruling_types):
        agg_verdict = VerdictType.SUPPORTED
    else:
        agg_verdict = VerdictType.NOT_ENOUGH_INFORMATION

    # Length-weighted confidence calculation
    total_len = sum(len(v.claim_text) for v in verdicts) or 1
    weighted_conf = sum(v.confidence * (len(v.claim_text) / total_len) for v in verdicts)
    weighted_conf = round(weighted_conf, 3)

    # Construct plain-English summary breakdown
    lines = [
        f"OVERALL VERDICT: {agg_verdict.value} (Confidence: {int(weighted_conf * 100)}%)",
        "",
        "CLAIM BREAKDOWN:",
    ]
    for v in verdicts:
        lines.append(f"• [{v.verdict.value}] \"{v.claim_text}\" (Confidence: {int(v.confidence * 100)}%) — {v.reasoning}")

    summary_text = "\n".join(lines)
    log_msg = format_audit_log(
        "Reporter",
        f"Synthesized aggregate verdict: {agg_verdict.value} across {len(verdicts)} claim(s)"
    )

    return {
        "aggregate_verdict": agg_verdict,
        "aggregate_confidence": weighted_conf,
        "audit_summary": summary_text,
        "verification_complete": True,
        "audit_trail": [log_msg],
        "error": None,
    }


def check_pending_node(state: VeritasState) -> Dict[str, Any]:
    """
    Pops the completed claim and prepares state for the next claim in queue.
    """
    remaining = list(state.get("pending_claim_ids", []))
    if remaining:
        completed_id = remaining.pop(0)

    next_claim = None
    if remaining:
        next_id = remaining[0]
        next_claim = next((c for c in state.get("atomic_claims", []) if c.claim_id == next_id), None)

    return {
        "pending_claim_ids": remaining,
        "current_claim": next_claim,
        "current_round": 0,
        "research_rounds": 0,
        "consensus_converged": False,
    }
