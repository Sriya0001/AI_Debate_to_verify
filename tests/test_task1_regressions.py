import pytest
import logging
from unittest.mock import patch, MagicMock

import numpy as np
from config import config
from models.claims import AtomicClaim
from models.agents import BeliefVector, AgentPosition
from models.evidence import EvidenceItem
from graph.state import initial_veritas_state
from graph.routing import route_after_consensus
from consensus.node import degroot_consensus_node
from eval.evaluate import run_single_llm_baseline, compute_metrics
from agents.judge import judge_node


# ── Task 1A & 1B Regressions: Routing, Consensus Convergence & Round Increment ─

def test_consensus_convergence_reflects_debate_disagreement():
    """
    Test 1A: DeGroot numerical solver converges, but if debaters disagree (> conflict_threshold),
    consensus_converged must be False so that debate/research loops remain reachable.
    """
    state = initial_veritas_state(input_text="Test statement", max_rounds=3)
    state["current_claim"] = AtomicClaim(
        claim_id="C1",
        claim_text="Test claim",
        claim_type="factual",
        entities=[],
        temporal_context="",
        verification_questions=[],
        source_span="",
    )
    # Debaters strongly disagree (disagreement = 0.8 > 0.35)
    state["belief_vectors"] = {
        "advocate": BeliefVector(supported=0.90, refuted=0.05, unknown=0.05),
        "skeptic": BeliefVector(supported=0.10, refuted=0.85, unknown=0.05),
        "neutral": BeliefVector(supported=0.40, refuted=0.40, unknown=0.20),
        "critic": BeliefVector(supported=0.30, refuted=0.30, unknown=0.40),
    }
    state["trust_weights"] = {
        "advocate": 0.25,
        "skeptic": 0.25,
        "neutral": 0.25,
        "critic": 0.25,
    }
    state["current_round"] = 0

    output = degroot_consensus_node(state)

    # DeGroot solver iterations converged, but debate consensus was NOT reached
    assert output["consensus_results"][-1].converged is True  # Numerical solver converged
    assert output["consensus_converged"] is False              # Debate consensus NOT reached
    assert output["current_round"] == 1                       # current_round properly incremented

    # Apply output to state and verify routing
    state.update(output)
    state["evidence_pool"] = [
        EvidenceItem(
            evidence_id=f"E{i}",
            claim_id="C1", query="q", url="http://test.com", title="T",
            snippet="S", source_domain="test.com"
        )
        for i in range(5)
    ]
    state["research_rounds"] = 2  # No more research needed

    next_route = route_after_consensus(state)
    # Must route to debate agents, NOT short-circuit to final_verifier!
    assert next_route == ["advocate", "skeptic", "neutral"]


def test_research_requests_not_blocked_by_consensus():
    """
    Test 1A: Critic research requests must route to autonomous_research
    when research rounds remain.
    """
    state = initial_veritas_state(input_text="Test statement", max_rounds=3, max_research_rounds=2)
    state["critic_research_requests"] = ["Need more evidence on primary source."]
    state["research_rounds"] = 0
    state["consensus_converged"] = False

    route = route_after_consensus(state)
    assert route == "autonomous_research"


def test_insufficient_evidence_routes_to_research():
    """
    Test 1A: Insufficient evidence pool (< min_evidence_count) routes to autonomous_research.
    """
    state = initial_veritas_state(input_text="Test statement", max_rounds=3, max_research_rounds=2)
    state["evidence_pool"] = []
    state["research_rounds"] = 0
    state["consensus_converged"] = False

    route = route_after_consensus(state)
    assert route == "autonomous_research"


def test_debate_loop_terminates_at_max_rounds():
    """
    Test 1B: Verify that current_round increments per consensus step
    and debate loops terminate at max_rounds even if agents still disagree.
    """
    state = initial_veritas_state(input_text="Test statement", max_rounds=2)
    state["current_claim"] = AtomicClaim(
        claim_id="C1",
        claim_text="Test claim",
        claim_type="factual",
        entities=[],
        temporal_context="",
        verification_questions=[],
        source_span="",
    )
    state["evidence_pool"] = [
        EvidenceItem(
            evidence_id=f"E{i}",
            claim_id="C1", query="q", url="http://test.com", title="T",
            snippet="S", source_domain="test.com"
        )
        for i in range(4)
    ]
    state["research_rounds"] = 2
    state["belief_vectors"] = {
        "advocate": BeliefVector(supported=0.90, refuted=0.05, unknown=0.05),
        "skeptic": BeliefVector(supported=0.10, refuted=0.85, unknown=0.05),
        "neutral": BeliefVector(supported=0.40, refuted=0.40, unknown=0.20),
        "critic": BeliefVector(supported=0.30, refuted=0.30, unknown=0.40),
    }

    # Round 1 execution:
    state["current_round"] = 0
    out1 = degroot_consensus_node(state)
    assert out1["current_round"] == 1
    state.update(out1)
    route1 = route_after_consensus(state)
    assert route1 == ["advocate", "skeptic", "neutral"]  # 1 < 2, continues

    # Round 2 execution:
    out2 = degroot_consensus_node(state)
    assert out2["current_round"] == 2
    state.update(out2)
    route2 = route_after_consensus(state)
    assert route2 == "final_verifier"  # 2 >= max_rounds (2), terminates!


# ── Task 1C Regressions: Single LLM Baseline Failure Recording ──────────────────

def test_run_single_llm_baseline_failure_recording():
    """
    Test 1C: Single LLM baseline must NOT fall back to 'SUPPORTED' on failure.
    It must retry, record failure explicitly, and return failed=True with verdict=None.
    """
    with patch("eval.evaluate.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API connection timeout")
        mock_get_llm.return_value = mock_llm

        res = run_single_llm_baseline("The moon is made of green cheese.")

        # Must NOT be "SUPPORTED"
        assert res["verdict"] is None
        assert res["confidence"] is None
        assert res["failed"] is True
        assert "API connection timeout" in res["error"]
        assert mock_llm.invoke.call_count == config.max_retries


def test_compute_metrics_excludes_failed_items():
    """
    Test 1C: compute_metrics must exclude failed items from accuracy/F1/ECE
    and report them separately in failed_count.
    """
    predictions = ["SUPPORTED", None, "REFUTED", "SUPPORTED"]
    ground_truths = ["SUPPORTED", "REFUTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
    confidences = [0.9, None, 0.85, 0.7]
    latencies = [1.0, 0.5, 1.2, 0.8]

    metrics = compute_metrics(predictions, ground_truths, confidences, latencies)

    # 3 valid items:
    # 1. P=SUPPORTED, G=SUPPORTED (correct)
    # 2. P=REFUTED, G=REFUTED (correct)
    # 3. P=SUPPORTED, G=NOT_ENOUGH_INFORMATION (incorrect)
    assert metrics["total_evaluated"] == 3
    assert metrics["failed_count"] == 1
    assert metrics["accuracy"] == round(2 / 3, 3)
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_compute_metrics_all_failed():
    """
    Test 1C: compute_metrics when all items fail should not crash or divide by zero.
    """
    metrics = compute_metrics([None, None], ["SUPPORTED", "REFUTED"], [None, None], [0.1, 0.1])
    assert metrics["accuracy"] == 0.0
    assert metrics["failed_count"] == 2
    assert metrics["total_evaluated"] == 0


# ── Task 1D Regressions: Judge Exception Fallback Logging & Audit Trail ─────────

def test_judge_fallback_logs_and_records_audit_trail(caplog):
    """
    Test 1D: When Judge LLM fails, it must retry, log the error,
    record the failure warning in the audit trail, and apply fallback weights.
    """
    state = initial_veritas_state(input_text="Test claim")
    state["current_claim"] = AtomicClaim(
        claim_id="C1",
        claim_text="Test claim text",
        claim_type="factual",
        entities=[],
        temporal_context="",
        verification_questions=[],
        source_span="",
    )

    with patch("agents.judge.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ValueError("Corrupted LLM JSON generation")
        mock_get_llm.return_value = mock_llm

        with caplog.at_level(logging.WARNING):
            result = judge_node(state)

        # 1. Fallback weights are provided
        assert "advocate" in result["trust_weights"]
        assert len(result["judge_evaluations"]) == 1

        # 2. Retry occurred max_retries times
        assert mock_llm.invoke.call_count == config.max_retries

        # 3. Warning / Error logged
        assert any("Corrupted LLM JSON generation" in record.message for record in caplog.records)

        # 4. Audit trail explicitly records failure warning
        audit_trail_text = " ".join(result["audit_trail"])
        assert "WARNING" in audit_trail_text
        assert "failed after" in audit_trail_text
        assert "applying fallback fixed trust weights" in audit_trail_text
