import numpy as np
from typing import Dict, List, Tuple


def build_trust_matrix(
    agent_names: List[str],
    judge_trust_weights: Dict[str, float],
    self_confidence_levels: Dict[str, float] = None,
) -> np.ndarray:
    """
    Constructs a row-stochastic trust matrix W where W[i, j] is the weight agent i
    assigns to agent j's belief.

    Args:
        agent_names: List of reasoning agent names (e.g. ['advocate', 'skeptic', 'neutral', 'critic'])
        judge_trust_weights: Normalized trust weights from the Judge Agent for each agent.
        self_confidence_levels: Optional dict mapping agent_name -> stubbornness/self-confidence (0.0 to 1.0).
            Default assigns 0.4 self-confidence and 0.6 external trust.

    Returns:
        A 2D numpy array of shape (N, N) where each row sums to 1.0.
    """
    n = len(agent_names)
    if n == 0:
        raise ValueError("Cannot construct trust matrix for empty agent list.")

    # Fallback default self-confidence (stubbornness factor)
    if self_confidence_levels is None:
        self_confidence_levels = {
            "advocate": 0.35,
            "skeptic": 0.35,
            "neutral": 0.20,
            "critic": 0.25,
        }

    # Normalize judge trust weights across provided agent names
    raw_weights = np.array([judge_trust_weights.get(name, 1.0 / n) for name in agent_names], dtype=float)
    total_raw = np.sum(raw_weights)
    if total_raw <= 0:
        base_weights = np.full(n, 1.0 / n, dtype=float)
    else:
        base_weights = raw_weights / total_raw

    W = np.zeros((n, n), dtype=float)

    for i, agent_i in enumerate(agent_names):
        # Stubbornness of agent i: how much it retains its own belief vs adopts group consensus
        alpha_i = self_confidence_levels.get(agent_i, 0.30)
        alpha_i = max(0.05, min(0.95, alpha_i))

        # Agent i retains (alpha_i) for self, and distributes (1 - alpha_i) according to Judge weights
        for j in range(n):
            if i == j:
                W[i, j] = alpha_i + (1.0 - alpha_i) * base_weights[j]
            else:
                W[i, j] = (1.0 - alpha_i) * base_weights[j]

        # Explicit row-stochastic normalization
        row_sum = np.sum(W[i, :])
        if row_sum > 0:
            W[i, :] = W[i, :] / row_sum

    return W


def validate_trust_matrix(W: np.ndarray, tol: float = 1e-4) -> bool:
    """
    Validates that:
    1. All entries 0 <= W[i, j] <= 1
    2. Every row sums to 1.0 within tolerance tol.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        return False
    if np.any(W < -tol) or np.any(W > 1.0 + tol):
        return False
    row_sums = np.sum(W, axis=1)
    return bool(np.all(np.abs(row_sums - 1.0) < tol))
