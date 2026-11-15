import concurrent.futures
import logging
from typing import Dict, Any, List
import numpy as np

from graph.state import VeritasState
from models.agents import BeliefVector, AgentPosition
from consensus.fcmad_math import (
    degroot_update,
    compute_max_disagreement,
    compute_final_pooling,
    calibrate_probabilities,
    check_abstain,
    should_stop,
    validate_belief,
)
from agents.fcmad_debater import create_debater_node
from utils.audit_trail import format_audit_log
from config import config

logger = logging.getLogger(__name__)


def fcmad_debaters_round0_node(state: VeritasState) -> Dict[str, Any]:
    """
    Executes all N debaters in parallel for Round 0.
    Input: claim + evidence pool, NO assigned stance.
    Collects initial belief vectors, records round 0 in per_round_beliefs and disagreement history.
    """
    num_d = getattr(config, "num_debaters", 3)
    current_claim = state.get("current_claim")
    claim_id = current_claim.claim_id if current_claim else "C1"

    all_positions: List[AgentPosition] = []
    belief_dict: Dict[str, BeliefVector] = {}
    audit_entries: List[str] = []
    recorded_failures: List[Dict[str, Any]] = list(state.get("recorded_failures", []))

    # Execute debaters concurrently
    debater_fns = [create_debater_node(i) for i in range(num_d)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_d) as executor:
        futures = [executor.submit(fn, state) for fn in debater_fns]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            all_positions.extend(res.get("agent_positions", []))
            belief_dict.update(res.get("belief_vectors", {}))
            audit_entries.extend(res.get("audit_trail", []))
            recorded_failures.extend(res.get("recorded_failures", []))

    # Order beliefs by debater id
    sorted_names = [f"debater_{i}" for i in range(num_d)]
    beliefs_list: List[List[float]] = []
    for name in sorted_names:
        if name in belief_dict:
            beliefs_list.append(belief_dict[name].to_list())
        else:
            beliefs_list.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    # Initial max disagreement E
    E = compute_max_disagreement(beliefs_list)
    log_msg = format_audit_log(
        "FC-MAD Engine",
        f"Round 0 initialized with {len(beliefs_list)} debaters. Initial max disagreement E={E:.4f}"
    )
    audit_entries.append(log_msg)

    per_round_beliefs = list(state.get("per_round_beliefs", []))
    per_round_beliefs.append(beliefs_list)

    dis_history = list(state.get("disagreement_history", []))
    dis_history.append(E)

    return {
        "agent_positions": all_positions,
        "belief_vectors": belief_dict,
        "per_round_beliefs": per_round_beliefs,
        "disagreement_history": dis_history,
        "current_round": 0,
        "recorded_failures": recorded_failures,
        "audit_trail": audit_entries,
        "error": None,
    }


def fcmad_debaters_revise_node(state: VeritasState) -> Dict[str, Any]:
    """
    Executes all N debaters in parallel for revision rounds (Round t >= 1).
    Input: claim + evidence + debate summary ONLY (not peers' numbers or credibilities).
    """
    num_d = getattr(config, "num_debaters", 3)
    current_round = state.get("current_round", 0)

    all_positions: List[AgentPosition] = []
    belief_dict: Dict[str, BeliefVector] = {}
    audit_entries: List[str] = []
    recorded_failures: List[Dict[str, Any]] = list(state.get("recorded_failures", []))

    debater_fns = [create_debater_node(i) for i in range(num_d)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_d) as executor:
        futures = [executor.submit(fn, state) for fn in debater_fns]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            all_positions.extend(res.get("agent_positions", []))
            belief_dict.update(res.get("belief_vectors", {}))
            audit_entries.extend(res.get("audit_trail", []))
            recorded_failures.extend(res.get("recorded_failures", []))

    log_msg = format_audit_log(
        "FC-MAD Engine",
        f"Round {current_round}: Debaters completed revision reasoning based on debate summary."
    )
    audit_entries.append(log_msg)

    return {
        "agent_positions": all_positions,
        "belief_vectors": belief_dict,
        "recorded_failures": recorded_failures,
        "audit_trail": audit_entries,
        "error": None,
    }


def fcmad_degroot_node(state: VeritasState) -> Dict[str, Any]:
    """
    Applies mathematical DeGroot update from consensus/fcmad_math.py:
        p_i^{new} = alpha * hat_p_i + (1 - alpha) * sum_{j != i} (c_j / sum_{k != i} c_k) * hat_p_j
    Increments the round counter in exactly this one place.
    """
    num_d = getattr(config, "num_debaters", 3)
    sorted_names = [f"debater_{i}" for i in range(num_d)]
    belief_dict = dict(state.get("belief_vectors", {}))

    beliefs_list: List[List[float]] = []
    for name in sorted_names:
        if name in belief_dict:
            beliefs_list.append(belief_dict[name].to_list())
        else:
            beliefs_list.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    cred_list = state.get("credibilities", [])
    if len(cred_list) != num_d:
        cred_arr = np.full(num_d, 1.0 / num_d)
    else:
        cred_arr = np.array(cred_list, dtype=float)

    # Perform DeGroot update
    updated_matrix = degroot_update(beliefs_list, cred_arr, alpha=config.alpha)

    # Update belief_vectors
    new_belief_dict: Dict[str, BeliefVector] = {}
    for idx, name in enumerate(sorted_names):
        vec = updated_matrix[idx]
        new_belief_dict[name] = BeliefVector(
            supported=float(round(vec[0], 4)),
            refuted=float(round(vec[1], 4)),
            unknown=float(round(vec[2], 4)),
        )

    # Compute new maximum disagreement E
    E = compute_max_disagreement(updated_matrix)

    # INCREMENT THE ROUND COUNTER IN EXACTLY ONE PLACE
    next_round = state.get("current_round", 0) + 1

    per_round_beliefs = list(state.get("per_round_beliefs", []))
    per_round_beliefs.append([list(v) for v in updated_matrix])

    dis_history = list(state.get("disagreement_history", []))
    dis_history.append(E)

    log_msg = format_audit_log(
        "DeGroot Update (FC-MAD)",
        f"Completed update for round {next_round}. New max disagreement E={E:.4f} (epsilon={config.epsilon:.3f})"
    )

    return {
        "current_round": next_round,
        "rounds_used": next_round,
        "belief_vectors": new_belief_dict,
        "per_round_beliefs": per_round_beliefs,
        "disagreement_history": dis_history,
        "audit_trail": [log_msg],
        "error": None,
    }


def fcmad_pool_node(state: VeritasState) -> Dict[str, Any]:
    """
    Computes final credibility-weighted pooling p*, label hat_y, and confidence:
        p* = sum_i c_i * p_i
        hat_y = argmax_k p^*_k
        raw confidence = max_k p^*_k
    Optionally applies temperature scaling calibration and checks abstain condition.
    """
    num_d = getattr(config, "num_debaters", 3)
    sorted_names = [f"debater_{i}" for i in range(num_d)]
    belief_dict = state.get("belief_vectors", {})

    beliefs_list: List[List[float]] = []
    for name in sorted_names:
        if name in belief_dict:
            beliefs_list.append(belief_dict[name].to_list())
        else:
            beliefs_list.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    cred_list = state.get("credibilities", [])
    if len(cred_list) != num_d:
        cred_arr = np.full(num_d, 1.0 / num_d)
    else:
        cred_arr = np.array(cred_list, dtype=float)

    p_star, label, raw_conf = compute_final_pooling(beliefs_list, cred_arr)

    # Calibration
    if config.calibration_enabled:
        p_cal, cal_conf = calibrate_probabilities(p_star, temperature=config.temperature_scaling_T)
        eff_conf = cal_conf
    else:
        p_cal, cal_conf = p_star, raw_conf
        eff_conf = raw_conf

    # Abstain check
    abstain_flag, _ = check_abstain(eff_conf, tau=config.abstain_threshold)

    log_msg = format_audit_log(
        "Pooling Engine (FC-MAD)",
        f"Consensus pooled distribution p*=[Supp={p_star[0]:.3f}, Ref={p_star[1]:.3f}, Unk={p_star[2]:.3f}] "
        f"-> Label: {label} | RawConf: {raw_conf:.1%} | CalConf: {cal_conf:.1%} (Abstain={abstain_flag})"
    )

    rounds_count = max(1, state.get("current_round", 0))
    return {
        "final_distribution": [float(round(v, 4)) for v in p_star],
        "raw_confidence": float(round(raw_conf, 4)),
        "calibrated_confidence": float(round(cal_conf, 4)),
        "abstain": abstain_flag,
        "rounds_used": rounds_count,
        "audit_trail": [log_msg],
        "error": None,
    }


def route_fcmad_stop(state: VeritasState) -> str:
    """
    Separate router for FC-MAD that stops ONLY on the epsilon rule or max rounds:
        E <= epsilon  -> "fcmad_pool"
        t >= M        -> "fcmad_pool"
        otherwise     -> "fcmad_summarizer"
    No shortcut flags.
    """
    dis_history = state.get("disagreement_history", [])
    current_E = dis_history[-1] if dis_history else 1.0
    current_round = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", config.max_rounds_fcmad)
    epsilon = config.epsilon

    stop, converged = should_stop(current_E, current_round, epsilon=epsilon, max_rounds=max_rounds)
    if stop:
        return "fcmad_pool"
    return "fcmad_summarizer"
