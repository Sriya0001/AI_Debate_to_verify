import pytest
import numpy as np
from itertools import permutations

from consensus.fcmad_math import (
    CLASSES,
    CLASS_TO_INDEX,
    validate_belief,
    total_variation_distance,
    initial_credibility,
    compute_credibilities,
    degroot_update,
    compute_max_disagreement,
    should_stop,
    compute_final_pooling,
    calibrate_probabilities,
    fit_temperature,
    check_abstain,
    compute_ece_10_bins,
    compute_brier_score,
)


# ── Required Test 1: All beliefs, p* and p_cal are valid probabilities ─────────

def test_1_all_beliefs_and_calibrations_are_valid_probabilities():
    """
    Spec Test 1: All beliefs, p* and p^{cal} must be valid probability distributions
    (non-negative, sum to 1.0, length 3) across multiple rounds of DeGroot update
    and varying calibration temperatures.
    """
    # Deterministic and stochastic initial beliefs for N=3 debaters
    initial_beliefs = np.array([
        [0.85, 0.10, 0.05],
        [0.20, 0.70, 0.10],
        [0.40, 0.30, 0.30],
    ])
    q_scores = [0.9, 0.8, 0.7]

    curr_beliefs = initial_beliefs.copy()

    # Simulate 5 sequential DeGroot update rounds
    for round_idx in range(5):
        cred = compute_credibilities(curr_beliefs, q_scores)
        assert np.all(cred > 0.0)
        assert abs(np.sum(cred) - 1.0) < 1e-6

        curr_beliefs = degroot_update(curr_beliefs, cred, alpha=0.7)

        # Every debater's belief must be a valid probability distribution
        for i in range(len(curr_beliefs)):
            b = curr_beliefs[i]
            assert b.shape == (3,)
            assert np.all(b >= 0.0)
            assert abs(float(np.sum(b)) - 1.0) < 1e-6

    # Verify p* validity
    cred = compute_credibilities(curr_beliefs, q_scores)
    p_star, label, raw_conf = compute_final_pooling(curr_beliefs, cred)

    assert p_star.shape == (3,)
    assert np.all(p_star >= 0.0)
    assert abs(float(np.sum(p_star)) - 1.0) < 1e-6
    assert label in CLASSES
    assert 1.0 / 3.0 <= raw_conf <= 1.0

    # Verify p^{cal} across multiple temperature values
    for T in [0.2, 0.5, 1.0, 1.5, 3.0, 10.0]:
        p_cal, cal_conf = calibrate_probabilities(p_star, temperature=T)
        assert p_cal.shape == (3,)
        assert np.all(p_cal >= 0.0)
        assert abs(float(np.sum(p_cal)) - 1.0) < 1e-6
        assert 1.0 / 3.0 <= cal_conf <= 1.0


# ── Required Test 2: Credibilities are positive and sum to 1 ───────────────────

def test_2_credibilities_are_positive_and_sum_to_one():
    """
    Spec Test 2: Credibilities c_i must always be strictly positive (due to +0.01)
    and sum to 1.0, even with extreme beliefs and quality scores (including q=0).
    """
    # Test case 1: Diverse beliefs, standard quality scores
    beliefs_1 = [
        [0.90, 0.05, 0.05],
        [0.85, 0.10, 0.05],
        [0.10, 0.85, 0.05],  # Outlier
    ]
    cred_1 = compute_credibilities(beliefs_1, quality_scores=[0.9, 0.85, 0.7])
    assert np.all(cred_1 > 0.0)
    assert abs(float(np.sum(cred_1)) - 1.0) < 1e-6
    # Majority debaters should have higher credibility than outlier
    assert cred_1[0] > cred_1[2]
    assert cred_1[1] > cred_1[2]

    # Test case 2: Extreme case where one debater has q_i = 0.0
    cred_zero_q = compute_credibilities(beliefs_1, quality_scores=[1.0, 1.0, 0.0])
    assert np.all(cred_zero_q > 0.0)  # Still positive due to 0.01 smoothing
    assert abs(float(np.sum(cred_zero_q)) - 1.0) < 1e-6
    assert cred_zero_q[2] < cred_zero_q[0]

    # Test case 3: Default quality scores (None -> all 1.0)
    cred_default = compute_credibilities(beliefs_1, quality_scores=None)
    assert np.all(cred_default > 0.0)
    assert abs(float(np.sum(cred_default)) - 1.0) < 1e-6

    # Test case 4: Larger swarm (N=5)
    beliefs_5 = [
        [0.7, 0.2, 0.1],
        [0.65, 0.25, 0.1],
        [0.68, 0.22, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
    ]
    cred_5 = compute_credibilities(beliefs_5)
    assert len(cred_5) == 5
    assert np.all(cred_5 > 0.0)
    assert abs(float(np.sum(cred_5)) - 1.0) < 1e-6


# ── Required Test 3: Identical beliefs give E=0 and p*=p_i ──────────────────────

def test_3_identical_beliefs_give_zero_disagreement_and_p_star_equals_p():
    """
    Spec Test 3: When all debaters hold identical beliefs:
    1. Maximum disagreement E must be 0.0
    2. DeGroot update leaves beliefs unchanged
    3. Consensus pooled distribution p* equals p_i
    """
    p_identical = [0.75, 0.15, 0.10]
    num_debaters = 4
    beliefs = [p_identical for _ in range(num_debaters)]

    # 1. Disagreement E must be 0.0
    E = compute_max_disagreement(beliefs)
    assert E == 0.0

    # 2. Credibilities are equal under uniform quality
    cred = compute_credibilities(beliefs)
    for c_i in cred:
        assert abs(c_i - 1.0 / num_debaters) < 1e-6

    # 3. DeGroot update preserves identical beliefs
    updated = degroot_update(beliefs, cred, alpha=0.7)
    for b in updated:
        assert np.allclose(b, p_identical, atol=1e-6)

    # 4. p* == p_i
    p_star, label, raw_conf = compute_final_pooling(beliefs, cred)
    assert np.allclose(p_star, p_identical, atol=1e-6)
    assert label == "SUPPORTED"
    assert abs(raw_conf - 0.75) < 1e-6

    # 5. Stop rule immediately declares consensus
    stop, converged = should_stop(E, round_num=1, epsilon=0.10, max_rounds=4)
    assert stop is True
    assert converged is True


# ── Required Test 4: Shuffling debater order does not change the result ────────

def test_4_shuffling_debater_order_invariance():
    """
    Spec Test 4: Permutation invariance:
    Reordering debaters (along with their quality scores) must produce:
    - The exact same max disagreement E
    - The exact same consensus pooled distribution p*
    - The exact same winning label and confidence
    - Credibilities permuted in the exact same order
    """
    beliefs = np.array([
        [0.85, 0.10, 0.05],
        [0.20, 0.75, 0.05],
        [0.45, 0.35, 0.20],
        [0.10, 0.10, 0.80],
    ])
    q_scores = np.array([0.95, 0.80, 0.85, 0.60])

    base_E = compute_max_disagreement(beliefs)
    base_cred = compute_credibilities(beliefs, q_scores)
    base_p_star, base_label, base_conf = compute_final_pooling(beliefs, base_cred)
    base_p_cal, base_cal_conf = calibrate_probabilities(base_p_star, temperature=1.2)

    # Test all 24 permutations of 4 debaters
    for perm in permutations(range(4)):
        perm = list(perm)
        perm_beliefs = beliefs[perm]
        perm_q = q_scores[perm]

        # 1. E must be invariant
        perm_E = compute_max_disagreement(perm_beliefs)
        assert abs(perm_E - base_E) < 1e-6

        # 2. Credibilities must match permuted base credibilities
        perm_cred = compute_credibilities(perm_beliefs, perm_q)
        assert np.allclose(perm_cred, base_cred[perm], atol=1e-6)

        # 3. p* must be identical
        perm_p_star, perm_label, perm_conf = compute_final_pooling(perm_beliefs, perm_cred)
        assert np.allclose(perm_p_star, base_p_star, atol=1e-6)
        assert perm_label == base_label
        assert abs(perm_conf - base_conf) < 1e-6

        # 4. Calibrated distribution must be identical
        perm_p_cal, perm_cal_conf = calibrate_probabilities(perm_p_star, temperature=1.2)
        assert np.allclose(perm_p_cal, base_p_cal, atol=1e-6)
        assert abs(perm_cal_conf - base_cal_conf) < 1e-6


# ── Required Test 5: A failed LLM call never produces a belief or a label ──────

def test_5_failed_call_never_produces_belief_or_label():
    """
    Spec Test 5: A failure must never become a verdict or valid belief.
    Corrupted inputs, negative values, non-simplex arrays, or missing values
    must be strictly rejected with ValueError and never yield a belief vector or label.
    """
    # 1. Negative probability
    with pytest.raises(ValueError, match="non-negative"):
        validate_belief([-0.1, 0.8, 0.3])

    # 2. Probabilities do not sum to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_belief([0.5, 0.5, 0.5])

    # 3. Invalid dimension (length != 3)
    with pytest.raises(ValueError, match="must have shape"):
        validate_belief([0.5, 0.5])

    with pytest.raises(ValueError, match="must have shape"):
        validate_belief([0.25, 0.25, 0.25, 0.25])

    # 4. NaN or Inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_belief([np.nan, 0.5, 0.5])

    # 5. compute_credibilities fails if any debater has invalid belief
    with pytest.raises(ValueError):
        compute_credibilities([
            [0.8, 0.1, 0.1],
            [0.9, 0.9, 0.9],  # Invalid sum
        ])

    # 6. compute_final_pooling fails on invalid inputs
    with pytest.raises(ValueError):
        compute_final_pooling(
            [[0.5, 0.5]],  # Invalid shape
            [1.0]
        )


# ── Edge-Case Test: N=2 Debaters ──────────────────────────────────────────────

def test_edge_case_n_equals_two():
    """
    Edge case: N=2 debaters.
    When N=2:
      bar_p_{-0} = p_1
      bar_p_{-1} = p_0
      s_0 = s_1 = 1 - d(p_0, p_1)
      DeGroot peer weight for each debater is 1.0 (only one peer).
    Verifies that no division by zero occurs and formulas remain exact.
    """
    p0 = [0.80, 0.15, 0.05]
    p1 = [0.20, 0.70, 0.10]
    beliefs = [p0, p1]

    # Distance d(p0, p1) = 0.5 * (|0.8 - 0.2| + |0.15 - 0.70| + |0.05 - 0.10|)
    #                    = 0.5 * (0.60 + 0.55 + 0.05) = 0.5 * 1.20 = 0.60
    dist = total_variation_distance(p0, p1)
    assert abs(dist - 0.60) < 1e-6

    # compute_credibilities for N=2
    # Equal quality: s_0 = s_1 = 0.40, so c_0 = c_1 = 0.5
    cred_equal = compute_credibilities(beliefs, quality_scores=[1.0, 1.0])
    assert len(cred_equal) == 2
    assert abs(cred_equal[0] - 0.5) < 1e-6
    assert abs(cred_equal[1] - 0.5) < 1e-6

    # Unequal quality: q_0 = 0.9, q_1 = 0.4
    # numerator 0: 0.40 * 0.9 + 0.01 = 0.36 + 0.01 = 0.37
    # numerator 1: 0.40 * 0.4 + 0.01 = 0.16 + 0.01 = 0.17
    # denom: 0.37 + 0.17 = 0.54
    # c_0 = 0.37 / 0.54 ≈ 0.6852, c_1 = 0.17 / 0.54 ≈ 0.3148
    cred_unequal = compute_credibilities(beliefs, quality_scores=[0.9, 0.4])
    assert cred_unequal[0] > cred_unequal[1]
    assert abs(cred_unequal[0] - (0.37 / 0.54)) < 1e-5
    assert abs(float(np.sum(cred_unequal)) - 1.0) < 1e-6

    # DeGroot update with N=2:
    # Each debater blends alpha * self + (1 - alpha) * other
    updated = degroot_update(beliefs, cred_equal, alpha=0.7)
    expected_p0_new = 0.7 * np.array(p0) + 0.3 * np.array(p1)
    expected_p1_new = 0.7 * np.array(p1) + 0.3 * np.array(p0)

    assert np.allclose(updated[0], expected_p0_new, atol=1e-6)
    assert np.allclose(updated[1], expected_p1_new, atol=1e-6)

    # Max disagreement E for N=2
    E = compute_max_disagreement(beliefs)
    assert abs(E - 0.60) < 1e-6

    # Pooling for N=2
    p_star, label, raw_conf = compute_final_pooling(beliefs, cred_unequal)
    assert p_star.shape == (3,)
    assert abs(float(np.sum(p_star)) - 1.0) < 1e-6
    assert label == "SUPPORTED"

    # Edge case: N < 2 raises error
    with pytest.raises(ValueError, match="At least 2 debaters"):
        compute_credibilities([[1.0, 0.0, 0.0]])


# ── Additional Spec Tests: Abstain, Calibration, and Metrics ───────────────────

def test_abstain_and_temperature_calibration():
    """
    Tests Section 6 (Calibration) & Section 7 (Abstain).
    """
    p_star = [0.55, 0.30, 0.15]
    # At T=1, conf is 0.55
    p_cal, conf = calibrate_probabilities(p_star, temperature=1.0)
    assert abs(conf - 0.55) < 1e-6

    # With tau = 0.60, conf 0.55 < 0.60 -> abstain = True, override = "uncertain"
    abstain, override = check_abstain(conf, tau=0.60)
    assert abstain is True
    assert override == "uncertain"

    # With higher confidence (e.g. 0.85) -> abstain = False
    abstain2, override2 = check_abstain(0.85, tau=0.60)
    assert abstain2 is False
    assert override2 is None


def test_metrics_ece_and_brier():
    """
    Tests Section 8 (ECE 10 bins & Brier score).
    """
    confidences = [0.9, 0.8, 0.7, 0.6]
    predictions = ["SUPPORTED", "SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
    ground_truths = ["SUPPORTED", "REFUTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]

    ece = compute_ece_10_bins(confidences, predictions, ground_truths)
    assert 0.0 <= ece <= 1.0

    probs = [
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ]
    brier = compute_brier_score(probs, ["SUPPORTED", "REFUTED", "REFUTED", "NOT_ENOUGH_INFORMATION"])
    assert brier >= 0.0
