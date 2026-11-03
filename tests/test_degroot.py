import pytest
import numpy as np
from consensus.trust import build_trust_matrix, validate_trust_matrix
from consensus.convergence import compute_max_disagreement, has_converged
from consensus.degroot import execute_degroot_matrix, run_degroot_consensus
from models.agents import BeliefVector


def test_trust_matrix_row_stochastic():
    agent_names = ["advocate", "skeptic", "neutral", "critic"]
    weights = {"advocate": 0.30, "skeptic": 0.35, "neutral": 0.25, "critic": 0.10}

    W = build_trust_matrix(agent_names, weights)

    assert W.shape == (4, 4)
    assert validate_trust_matrix(W)
    for row in W:
        assert np.isclose(np.sum(row), 1.0, atol=1e-5)
        for val in row:
            assert 0.0 <= val <= 1.0


def test_degroot_matrix_multiplication():
    # 2-agent equal trust matrix
    W = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    # Agent 1 believes [1, 0, 0], Agent 2 believes [0, 1, 0]
    X0 = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])

    final_X, history, converged = execute_degroot_matrix(W, X0, max_iterations=10, convergence_threshold=1e-4)

    assert converged is True
    # In one step it should reach [0.5, 0.5, 0.0]
    expected = np.array([[0.5, 0.5, 0.0],
                         [0.5, 0.5, 0.0]])
    np.testing.assert_allclose(final_X, expected, atol=1e-4)


def test_degroot_full_consensus():
    beliefs = {
        "advocate": BeliefVector(supported=0.80, refuted=0.10, unknown=0.10),
        "skeptic": BeliefVector(supported=0.20, refuted=0.65, unknown=0.15),
        "neutral": BeliefVector(supported=0.55, refuted=0.20, unknown=0.25),
        "critic": BeliefVector(supported=0.40, refuted=0.30, unknown=0.30),
    }
    weights = {
        "advocate": 0.30,
        "skeptic": 0.35,
        "neutral": 0.25,
        "critic": 0.10,
    }

    result = run_degroot_consensus(
        claim_id="C1",
        round_number=1,
        agent_beliefs=beliefs,
        judge_trust_weights=weights,
        convergence_threshold=1e-4,
        max_iterations=50,
    )

    assert result.converged is True
    assert result.iterations_run >= 1
    assert len(result.history) > 1

    # Final consensus must satisfy probability axioms
    fc = result.final_consensus
    total = fc.supported + fc.refuted + fc.unknown
    assert np.isclose(total, 1.0, atol=1e-3)
    assert 0.0 <= fc.supported <= 1.0
    assert 0.0 <= fc.refuted <= 1.0
    assert 0.0 <= fc.unknown <= 1.0

    # Max disagreement at end must be smaller than at start
    initial_disagreement = result.history[0].max_disagreement
    final_disagreement = result.history[-1].max_disagreement
    assert final_disagreement <= initial_disagreement


def test_belief_vector_auto_normalization():
    # If slight drift occurs, BeliefVector auto-normalizes
    bv = BeliefVector(supported=0.7, refuted=0.2, unknown=0.2) # sum=1.1
    assert np.isclose(bv.supported + bv.refuted + bv.unknown, 1.0, atol=1e-3)
