"""
FC-MAD Mathematical Module (docs/FCMAD_MATH.md)
Pure, LLM-free mathematical core implementing all formulas from docs/FCMAD_MATH.md.

Single source of truth for the 'fcmad' mode:
1. Disagreement (Total Variation): d(p, q) = 0.5 * sum_k |p_k - q_k| in [0, 1]
2. Credibility: s_i = 1 - d(p_i, bar_p_{-i}), c_i = (s_i * q_i + 0.01) / sum_k (s_k * q_k + 0.01)
3. DeGroot update: p_i^{new} = alpha * p_i + (1 - alpha) * sum_{j != i} (c_j / sum_{k != i} c_k) * p_j
4. Stop rule: E = max_{i, j} d(p_i, p_j) <= epsilon or t = M
5. Final pooling: p* = sum_i c_i * p_i, hat_y = argmax_k p^*_k, raw confidence = max_k p^*_k
6. Calibration: p^{cal}_k = (p^*_k)^{1/T} / sum_l (p^*_l)^{1/T}, calibrated confidence = max_k p^{cal}_k
7. Abstain: if calibrated confidence < tau, output 'uncertain'
8. Metrics: ECE (10 bins), Brier score
"""

import numpy as np
from typing import List, Tuple, Optional, Sequence, Union, Dict, Any


# Class definitions according to FCMAD_MATH.md
# Classes: S (Supported), R (Refuted), N (Not-Enough-Info)
CLASSES: List[str] = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
SHORT_CLASSES: List[str] = ["S", "R", "N"]
CLASS_TO_INDEX: Dict[str, int] = {
    "SUPPORTED": 0,
    "REFUTED": 1,
    "NOT_ENOUGH_INFORMATION": 2,
    "S": 0,
    "R": 1,
    "N": 2,
}


def validate_belief(p: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """
    Validates that a belief vector has length 3, non-negative entries, and sums to 1.0.

    Raises:
        ValueError: If dimension is invalid, entries are negative, NaN, or sum is not 1.0.
    """
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != 3:
        raise ValueError(f"Belief vector must have shape (3,), got {arr.shape}")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Belief vector contains NaN or Inf values")
    if np.any(arr < 0.0):
        raise ValueError(f"Belief probabilities must be non-negative, got {arr}")
    total = float(np.sum(arr))
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Belief vector must sum to 1.0, got sum={total:.6f} for {arr}")
    # Return normalized float vector
    return arr / total


def total_variation_distance(
    p: Union[np.ndarray, Sequence[float]],
    q: Union[np.ndarray, Sequence[float]],
) -> float:
    """
    Computes Total Variation Distance between two categorical distributions p and q:
        d(p, q) = 0.5 * sum_k |p_k - q_k| in [0, 1]
    """
    p_arr = validate_belief(p)
    q_arr = validate_belief(q)
    return float(0.5 * np.sum(np.abs(p_arr - q_arr)))


def initial_credibility(num_debaters: int) -> np.ndarray:
    """
    Returns initial credibility distribution: c_i = 1/N for each debater.
    """
    if num_debaters < 2:
        raise ValueError("num_debaters must be at least 2")
    return np.full(num_debaters, 1.0 / num_debaters, dtype=float)


def compute_credibilities(
    beliefs: Union[np.ndarray, Sequence[Sequence[float]]],
    quality_scores: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> np.ndarray:
    """
    Computes credibility c_i for each debater:
        bar_p_{-i} = 1/(N - 1) * sum_{j != i} p_j
        s_i = 1 - d(p_i, bar_p_{-i})
        c_i = (s_i * q_i + 0.01) / sum_k (s_k * q_k + 0.01)

    Args:
        beliefs: Array of shape (N, 3) where each row is a valid belief vector.
        quality_scores: Optional array of length N with Judge quality scores q_i in [0, 1].
                        Defaults to q_i = 1.0 for all debaters (Judge switched off).

    Returns:
        Credibility array of length N, non-negative, summing to 1.0.
    """
    beliefs_arr = np.array([validate_belief(b) for b in beliefs], dtype=float)
    n = beliefs_arr.shape[0]
    if n < 2:
        raise ValueError(f"At least 2 debaters are required to compute peer credibility, got N={n}")

    if quality_scores is None:
        q = np.ones(n, dtype=float)
    else:
        q = np.asarray(quality_scores, dtype=float)
        if q.shape != (n,):
            raise ValueError(f"quality_scores must have shape ({n},), got {q.shape}")
        if np.any(q < 0.0) or np.any(q > 1.0):
            raise ValueError(f"quality_scores must be in [0, 1], got {q}")

    s = np.zeros(n, dtype=float)
    for i in range(n):
        # Average of the *others*
        other_mask = np.ones(n, dtype=bool)
        other_mask[i] = False
        bar_p_minus_i = np.sum(beliefs_arr[other_mask], axis=0) / (n - 1)
        # Ensure bar_p_minus_i sums to 1
        bar_p_minus_i = bar_p_minus_i / np.sum(bar_p_minus_i)

        # Closeness s_i
        dist = total_variation_distance(beliefs_arr[i], bar_p_minus_i)
        s[i] = max(0.0, min(1.0, 1.0 - dist))

    # Credibility formula: c_i = (s_i * q_i + 0.01) / sum_k (s_k * q_k + 0.01)
    numerator = s * q + 0.01
    denominator = float(np.sum(numerator))
    c = numerator / denominator
    return c


def degroot_update(
    beliefs: Union[np.ndarray, Sequence[Sequence[float]]],
    credibilities: Union[np.ndarray, Sequence[float]],
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Applies one step of DeGroot update per round:
        p_i^{new} = alpha * p_i + (1 - alpha) * sum_{j != i} (c_j / sum_{k != i} c_k) * p_j

    Args:
        beliefs: Array of shape (N, 3).
        credibilities: Array of shape (N,).
        alpha: Stubbornness parameter in [0, 1] (default 0.7).

    Returns:
        Updated beliefs of shape (N, 3), valid probability distributions.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    beliefs_arr = np.array([validate_belief(b) for b in beliefs], dtype=float)
    c_arr = np.asarray(credibilities, dtype=float)
    n = beliefs_arr.shape[0]
    if c_arr.shape != (n,):
        raise ValueError(f"credibilities must have shape ({n},), got {c_arr.shape}")

    updated_beliefs = np.zeros_like(beliefs_arr)

    for i in range(n):
        other_mask = np.ones(n, dtype=bool)
        other_mask[i] = False

        c_others = c_arr[other_mask]
        sum_c_others = float(np.sum(c_others))

        if sum_c_others > 0:
            peer_weights = c_others / sum_c_others
            peer_mix = np.sum(beliefs_arr[other_mask] * peer_weights[:, np.newaxis], axis=0)
        else:
            peer_mix = beliefs_arr[i]

        new_pi = alpha * beliefs_arr[i] + (1.0 - alpha) * peer_mix
        # Enforce simplex constraint
        new_pi = np.maximum(0.0, new_pi)
        sum_new = np.sum(new_pi)
        if sum_new > 0:
            new_pi = new_pi / sum_new
        else:
            new_pi = np.full(3, 1.0 / 3.0)

        updated_beliefs[i] = new_pi

    return updated_beliefs


def compute_max_disagreement(beliefs: Union[np.ndarray, Sequence[Sequence[float]]]) -> float:
    """
    Computes maximum pairwise Total Variation disagreement across all debaters:
        E = max_{i, j} d(p_i, p_j)
    """
    beliefs_arr = np.array([validate_belief(b) for b in beliefs], dtype=float)
    n = beliefs_arr.shape[0]
    if n <= 1:
        return 0.0

    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = total_variation_distance(beliefs_arr[i], beliefs_arr[j])
            if dist > max_dist:
                max_dist = dist

    return float(round(max_dist, 6))


def should_stop(
    E: float,
    round_num: int,
    epsilon: float = 0.10,
    max_rounds: int = 4,
) -> Tuple[bool, bool]:
    """
    Stop rule:
        Stop when E <= epsilon (consensus) or round_num >= max_rounds (record 'no consensus').

    Returns:
        (should_stop: bool, converged: bool)
    """
    if E <= epsilon:
        return True, True
    if round_num >= max_rounds:
        return True, False
    return False, False


def compute_final_pooling(
    beliefs: Union[np.ndarray, Sequence[Sequence[float]]],
    credibilities: Union[np.ndarray, Sequence[float]],
) -> Tuple[np.ndarray, str, float]:
    """
    Computes credibility-weighted consensus belief p*, predicted class hat_y, and raw confidence:
        p* = sum_i c_i * p_i
        hat_y = argmax_k p^*_k
        raw confidence = max_k p^*_k

    Returns:
        (p_star, label_name, raw_confidence)
    """
    beliefs_arr = np.array([validate_belief(b) for b in beliefs], dtype=float)
    c_arr = np.asarray(credibilities, dtype=float)
    n = beliefs_arr.shape[0]
    if c_arr.shape != (n,):
        raise ValueError(f"credibilities must have shape ({n},), got {c_arr.shape}")

    c_sum = float(np.sum(c_arr))
    if c_sum <= 0:
        c_normalized = np.full(n, 1.0 / n)
    else:
        c_normalized = c_arr / c_sum

    p_star = np.sum(beliefs_arr * c_normalized[:, np.newaxis], axis=0)
    p_star = np.maximum(0.0, p_star)
    p_star = p_star / np.sum(p_star)

    winning_idx = int(np.argmax(p_star))
    label = CLASSES[winning_idx]
    raw_confidence = float(p_star[winning_idx])

    return p_star, label, raw_confidence


def calibrate_probabilities(
    p_star: Union[np.ndarray, Sequence[float]],
    temperature: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Applies temperature scaling calibration:
        p^{cal}_k = (p^*_k)^{1/T} / sum_l (p^*_l)^{1/T}
        calibrated confidence = max_k p^{cal}_k

    Args:
        p_star: Consensus belief vector [p_S, p_R, p_N].
        temperature: Temperature parameter T > 0 (T=1 is identity).

    Returns:
        (p_cal, calibrated_confidence)
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature T must be strictly positive, got {temperature}")

    p = validate_belief(p_star)
    if abs(temperature - 1.0) < 1e-9:
        return p.copy(), float(np.max(p))

    # Numerical stability with power
    safe_p = np.maximum(p, 1e-12)
    inv_t = 1.0 / temperature
    powered = safe_p ** inv_t
    denom = float(np.sum(powered))
    p_cal = powered / denom
    p_cal = np.maximum(0.0, p_cal)
    p_cal = p_cal / np.sum(p_cal)

    calibrated_confidence = float(np.max(p_cal))
    return p_cal, calibrated_confidence


def fit_temperature(
    p_stars: Sequence[Union[np.ndarray, Sequence[float]]],
    ground_truth_labels: Sequence[Union[int, str]],
    initial_T: float = 1.0,
) -> float:
    """
    Fits calibration temperature T on dev split by minimizing negative log-likelihood:
        T = argmin_T -sum_n ln p^{cal}_n[y_n]
    """
    if len(p_stars) == 0:
        return 1.0

    valid_p_stars = [validate_belief(p) for p in p_stars]
    label_indices = []
    for gt in ground_truth_labels:
        if isinstance(gt, (int, np.integer)):
            label_indices.append(int(gt))
        elif isinstance(gt, str):
            label_indices.append(CLASS_TO_INDEX[gt.upper()])
        else:
            raise ValueError(f"Unknown ground truth label format: {gt}")

    def nll(T_val: float) -> float:
        if T_val <= 1e-3:
            return 1e9
        total_loss = 0.0
        for p_s, y_idx in zip(valid_p_stars, label_indices):
            p_cal, _ = calibrate_probabilities(p_s, temperature=T_val)
            prob_y = max(p_cal[y_idx], 1e-12)
            total_loss -= np.log(prob_y)
        return float(total_loss)

    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        if res.success:
            return float(round(res.x, 4))
    except Exception:
        pass

    # Simple grid search fallback if scipy not available or failed
    best_t = initial_T
    best_loss = nll(initial_T)
    for candidate_t in np.linspace(0.1, 5.0, 100):
        loss = nll(float(candidate_t))
        if loss < best_loss:
            best_loss = loss
            best_t = float(candidate_t)

    return float(round(best_t, 4))


def check_abstain(
    calibrated_confidence: float,
    tau: float = 0.60,
) -> Tuple[bool, Optional[str]]:
    """
    Abstain rule (FCMAD_MATH.md Section 7):
        If calibrated confidence < tau, output 'uncertain'.
        This is separate from NEI (insufficient evidence): abstain means
        the system is unsure of its own verdict.

    Returns:
        (abstain_flag: bool, label_override: Optional[str])
    """
    if calibrated_confidence < tau:
        return True, "uncertain"
    return False, None


def compute_ece_10_bins(
    confidences: Sequence[float],
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> float:
    """
    Expected Calibration Error (ECE) over 10 equal-width bins (FCMAD_MATH.md Section 8):
        ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
    """
    if not confidences or len(confidences) != len(predictions):
        return 0.0

    n = len(confidences)
    bin_boundaries = np.linspace(0.0, 1.0, 11)  # 10 bins
    ece = 0.0

    for i in range(10):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        indices = [
            idx for idx, c in enumerate(confidences)
            if (bin_lower <= c < bin_upper) or (i == 9 and bin_lower <= c <= bin_upper)
        ]

        if not indices:
            continue

        bin_conf = float(np.mean([confidences[idx] for idx in indices]))
        bin_acc = float(np.mean([1.0 if predictions[idx] == ground_truths[idx] else 0.0 for idx in indices]))
        bin_weight = len(indices) / n

        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def compute_brier_score(
    probabilities: Sequence[Union[np.ndarray, Sequence[float]]],
    ground_truth_indices: Sequence[Union[int, str]],
) -> float:
    """
    Brier Score (FCMAD_MATH.md Section 8):
        Brier = (1 / n) * sum_n sum_k (p_{n, k} - 1[y_n = k])^2
    """
    if not probabilities or len(probabilities) != len(ground_truth_indices):
        return 0.0

    n = len(probabilities)
    total_brier = 0.0

    for p, gt in zip(probabilities, ground_truth_indices):
        p_arr = validate_belief(p)
        if isinstance(gt, str):
            gt_idx = CLASS_TO_INDEX[gt.upper()]
        else:
            gt_idx = int(gt)

        one_hot = np.zeros(3, dtype=float)
        one_hot[gt_idx] = 1.0

        total_brier += float(np.sum((p_arr - one_hot) ** 2))

    return float(round(total_brier / n, 4))
