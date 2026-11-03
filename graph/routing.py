from typing import List, Dict, Any, Union
from graph.state import VeritasState
from models.evidence import EvidenceItem, SourceScore
from models.agents import BeliefVector
from consensus.convergence import compute_max_disagreement
from config import config
from utils.audit_trail import format_audit_log


def route_after_consensus(state: VeritasState) -> Union[str, List[str]]:
    """
    Autonomous Router evaluating state after DeGroot consensus.
    Decides whether to:
      1. Trigger additional research ("autonomous_research")
      2. Trigger another debate round (["advocate", "skeptic", "neutral"])
      3. Proceed to Final Verification ("final_verifier")
    """
    evidence_items = state.get("evidence_pool", [])
    belief_vectors = state.get("belief_vectors", {})
    source_scores = state.get("source_scores", {})
    critic_requests = state.get("critic_research_requests", [])
    consensus_converged = state.get("consensus_converged", False)

    current_round = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", config.max_debate_rounds)
    research_rounds = state.get("research_rounds", 0)
    max_research_rounds = state.get("max_research_rounds", config.max_research_rounds)

    # 1. If consensus has already converged, proceed directly to final verifier
    if consensus_converged:
        return "final_verifier"

    # 2. Check if Socratic Critic requested additional research
    if critic_requests and research_rounds < max_research_rounds:
        return "autonomous_research"

    # 3. Check if evidence is critically insufficient
    if len(evidence_items) < config.min_evidence_count and research_rounds < max_research_rounds:
        return "autonomous_research"

    # 4. Check for high-credibility conflicting sources
    has_conflict = _detect_source_conflict(evidence_items, source_scores)
    if has_conflict and research_rounds < max_research_rounds:
        return "autonomous_research"

    # 5. Check for high disagreement between agents if rounds remain
    if belief_vectors:
        beliefs_matrix = [bv.to_list() for bv in belief_vectors.values()]
        import numpy as np
        disagreement = compute_max_disagreement(np.array(beliefs_matrix))
    else:
        disagreement = 0.0

    if disagreement > config.conflict_threshold and current_round < max_rounds:
        # Fan-out to debate reasoning agents for another round
        return ["advocate", "skeptic", "neutral"]

    # 6. Default / Max rounds reached -> Final Verification
    return "final_verifier"


def route_after_verifier(state: VeritasState) -> str:
    """
    Decides whether there are more claims pending in the queue,
    or whether to proceed to final aggregate reporting.
    """
    remaining = state.get("pending_claim_ids", [])
    if remaining:
        return "check_pending"
    return "reporter"


def route_after_pending(state: VeritasState) -> str:
    """
    After advancing to the next claim, routes back to planner.
    If no more claims, routes to reporter.
    """
    if state.get("pending_claim_ids"):
        return "planner"
    return "reporter"


def _detect_source_conflict(evidence: List[EvidenceItem], source_scores: Dict[str, SourceScore]) -> bool:
    """
    Detects if two high-credibility sources take directly contradictory stances.
    """
    high_supp = False
    high_contra = False

    for e in evidence:
        score_obj = source_scores.get(e.source_domain)
        cred = score_obj.credibility_score if score_obj else 0.7
        if cred >= config.source_credibility_threshold:
            if e.supports_claim is True:
                high_supp = True
            elif e.supports_claim is False:
                high_contra = True

    return high_supp and high_contra
