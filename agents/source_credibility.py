from typing import Dict, Any, List
from graph.state import VeritasState
from models.evidence import SourceScore, EvidenceItem
from tools.source_analysis import analyze_source_credibility
from utils.audit_trail import format_audit_log


def source_credibility_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Evaluates the credibility, authority level, independence,
    and potential bias of all sources retrieved in the evidence pool.
    """
    evidence_items: List[EvidenceItem] = state.get("evidence_pool", [])
    current_claim = state.get("current_claim")
    claim_id = current_claim.claim_id if current_claim else "C1"

    updated_scores: Dict[str, SourceScore] = dict(state.get("source_scores", {}))
    newly_evaluated = 0

    for item in evidence_items:
        domain = item.source_domain
        if domain not in updated_scores:
            score_obj = analyze_source_credibility(url=item.url, domain=domain)
            updated_scores[domain] = score_obj
            newly_evaluated += 1

    high_authority_count = sum(1 for s in updated_scores.values() if s.credibility_score >= 0.85)
    log_msg = format_audit_log(
        "Source Credibility Agent",
        f"Evaluated {len(updated_scores)} source domain(s) ({high_authority_count} high-authority) for claim {claim_id}"
    )

    return {
        "source_scores": updated_scores,
        "audit_trail": [log_msg],
        "error": None,
    }
