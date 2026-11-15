import numpy as np
from typing import Dict, Any
from graph.state import VeritasState
from models.agents import BeliefVector
from consensus.degroot import run_degroot_consensus
from consensus.convergence import compute_max_disagreement
from config import config
from utils.audit_trail import format_audit_log


def degroot_consensus_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Executes mathematical DeGroot consensus on agent belief vectors
    using Judge-assigned trust weights.
    """
    current_claim = state.get("current_claim")
    claim_id = current_claim.claim_id if current_claim else "C1"

    beliefs: Dict[str, BeliefVector] = dict(state.get("belief_vectors", {}))
    trust_weights: Dict[str, float] = dict(state.get("trust_weights", {}))

    # Ensure all 4 agents have belief vectors
    default_names = ["advocate", "skeptic", "neutral", "critic"]
    for name in default_names:
        if name not in beliefs:
            beliefs[name] = BeliefVector(supported=1.0/3.0, refuted=1.0/3.0, unknown=1.0/3.0)
        if name not in trust_weights:
            trust_weights[name] = 1.0 / len(default_names)

    # Execute Python mathematical DeGroot consensus
    result = run_degroot_consensus(
        claim_id=claim_id,
        round_number=state.get("current_round", 0) + 1,
        agent_beliefs=beliefs,
        judge_trust_weights=trust_weights,
        convergence_threshold=config.convergence_threshold,
        max_iterations=config.max_consensus_iterations,
    )

    # Check whether debate agents have reached consensus based on disagreement threshold
    beliefs_matrix = np.array([bv.to_list() for bv in beliefs.values()])
    disagreement = compute_max_disagreement(beliefs_matrix)
    consensus_converged = bool(result.converged and (disagreement <= config.conflict_threshold))

    fc = result.final_consensus
    log_msg = format_audit_log(
        "DeGroot Consensus Engine",
        f"SolverConverged={result.converged} (Disagreement={disagreement:.3f}, ConsensusReached={consensus_converged}) "
        f"in {result.iterations_run} iteration(s). "
        f"Consensus: P(Supp)={fc.supported:.2f}, P(Ref)={fc.refuted:.2f}, P(Unk)={fc.unknown:.2f}"
    )

    consensus_list = list(state.get("consensus_results", []))
    consensus_list.append(result)

    return {
        "consensus_results": consensus_list,
        "consensus_converged": consensus_converged,
        "trust_matrix": result.trust_matrix,
        "current_round": state.get("current_round", 0) + 1,
        "audit_trail": [log_msg],
        "error": None,
    }
