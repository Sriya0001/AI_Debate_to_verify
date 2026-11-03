import numpy as np
from typing import Dict, List, Tuple
from models.agents import BeliefVector
from models.consensus import DeGrootIteration, ConsensusResult
from consensus.trust import build_trust_matrix, validate_trust_matrix
from consensus.convergence import compute_max_disagreement, has_converged, compute_stationary_distribution


def execute_degroot_matrix(
    W: np.ndarray,
    X0: np.ndarray,
    max_iterations: int = 50,
    convergence_threshold: float = 1e-4,
) -> Tuple[np.ndarray, List[np.ndarray], bool]:
    """
    Core mathematical execution of the DeGroot model:
        X(t+1) = W * X(t)
    
    Args:
        W: Row-stochastic trust matrix of shape (N, N).
        X0: Initial belief matrix of shape (N, C) where C=3 (supported, refuted, unknown).
        max_iterations: Maximum iteration cutoff.
        convergence_threshold: Maximum element-wise change to declare convergence.

    Returns:
        (final_X, history_matrices, converged_flag)
    """
    if not validate_trust_matrix(W):
        # Renormalize rows if slight floating point drift exists
        row_sums = np.sum(W, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums

    history = [X0.copy()]
    curr_X = X0.copy()
    converged = False

    for t in range(max_iterations):
        # Matrix multiplication X(t+1) = W * X(t)
        next_X = np.matmul(W, curr_X)

        # Enforce non-negativity and row simplex constraint (probabilities sum to 1)
        next_X = np.maximum(0.0, next_X)
        sums = np.sum(next_X, axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        next_X = next_X / sums

        history.append(next_X.copy())

        if has_converged(curr_X, next_X, tol=convergence_threshold):
            curr_X = next_X
            converged = True
            break

        curr_X = next_X

    return curr_X, history, converged


def run_degroot_consensus(
    claim_id: str,
    round_number: int,
    agent_beliefs: Dict[str, BeliefVector],
    judge_trust_weights: Dict[str, float],
    convergence_threshold: float = 1e-4,
    max_iterations: int = 50,
) -> ConsensusResult:
    """
    Executes full DeGroot consensus on the agent team's belief vectors.

    Args:
        claim_id: ID of the claim being debated.
        round_number: Current debate round.
        agent_beliefs: Dict mapping agent name to BeliefVector.
        judge_trust_weights: Dict mapping agent name to trust weight.
        convergence_threshold: Convergence epsilon.
        max_iterations: Max iterations before stopping.

    Returns:
        Structured ConsensusResult with full convergence history.
    """
    agent_names = list(agent_beliefs.keys())
    if not agent_names:
        raise ValueError("Cannot run DeGroot consensus with zero agents.")

    # 1. Build trust matrix W
    W = build_trust_matrix(agent_names, judge_trust_weights)

    # 2. Build initial belief matrix X0
    X0 = np.array([agent_beliefs[name].to_list() for name in agent_names], dtype=float)

    # 3. Execute DeGroot dynamics
    final_X, history_matrices, converged = execute_degroot_matrix(
        W=W,
        X0=X0,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
    )

    # 4. Compute stationary distribution of W
    pi = compute_stationary_distribution(W)

    # 5. Build DeGrootIteration history objects
    iteration_records: List[DeGrootIteration] = []
    for step_idx, X_step in enumerate(history_matrices):
        beliefs_dict = {
            agent_names[i]: [round(float(v), 4) for v in X_step[i]]
            for i in range(len(agent_names))
        }
        mean_b = [round(float(v), 4) for v in np.mean(X_step, axis=0)]
        max_dis = compute_max_disagreement(X_step)

        iteration_records.append(
            DeGrootIteration(
                iteration=step_idx,
                agent_beliefs=beliefs_dict,
                mean_belief=mean_b,
                max_disagreement=max_dis,
            )
        )

    # 6. Final consensus belief: asymptotic stationary weighted sum
    # pi * X is the true mathematical consensus under DeGroot
    consensus_raw = np.dot(pi, final_X)
    consensus_raw = np.maximum(0.0, consensus_raw)
    total_c = np.sum(consensus_raw)
    if total_c > 0:
        consensus_raw = consensus_raw / total_c
    else:
        consensus_raw = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    final_belief = BeliefVector(
        supported=float(consensus_raw[0]),
        refuted=float(consensus_raw[1]),
        unknown=float(consensus_raw[2]),
    )

    return ConsensusResult(
        claim_id=claim_id,
        round_number=round_number,
        converged=converged,
        iterations_run=len(history_matrices) - 1,
        final_consensus=final_belief,
        history=iteration_records,
        trust_matrix=[[round(float(val), 4) for val in row] for row in W],
        agent_names=agent_names,
    )
