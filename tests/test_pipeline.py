import pytest
import numpy as np

from models.claims import AtomicClaim
from models.evidence import EvidenceItem, ResearchTask
from models.agents import BeliefVector, AgentPosition, JudgeEvaluation
from models.verdict import VerdictType, FinalVerdict
from tools.evidence import evidence_pool
from tools.source_analysis import analyze_source_credibility
from graph.routing import route_after_consensus, route_after_verifier
from graph.state import initial_veritas_state
from runner import run_audit


def test_claim_decomposition_model():
    claim = AtomicClaim(
        claim_id="C1",
        claim_text="The Eiffel Tower was completed in 1889.",
        claim_type="temporal",
        entities=["Eiffel Tower"],
        temporal_context="1889",
        verification_questions=["When was construction finished?"],
        source_span="completed in 1889",
    )
    assert claim.claim_id == "C1"
    assert "Eiffel Tower" in claim.entities
    assert claim.claim_type == "temporal"


def test_evidence_indexing_and_deduplication():
    pool = evidence_pool
    item1 = pool.add_item(
        claim_id="C1",
        title="Official Records",
        url="https://www.toureiffel.paris/en/history",
        snippet="Built in 1889",
        source_domain="toureiffel.paris",
        supports_claim=True,
    )
    assert item1.evidence_id.startswith("E")

    # Adding same URL returns identical item
    item2 = pool.add_item(
        claim_id="C1",
        title="Official Records Copy",
        url="https://www.toureiffel.paris/en/history",
        snippet="Built in 1889",
        source_domain="toureiffel.paris",
    )
    assert item1.evidence_id == item2.evidence_id


def test_source_credibility_hierarchy():
    # Level 1: Gov
    gov_score = analyze_source_credibility("https://www.sec.gov/edgar")
    assert gov_score.authority_level == 1
    assert gov_score.credibility_score >= 0.95

    # Level 3: Academic
    nature_score = analyze_source_credibility("https://www.nature.com/articles/123")
    assert nature_score.authority_level == 3
    assert nature_score.credibility_score >= 0.90

    # Level 4: News
    reuters_score = analyze_source_credibility("https://www.reuters.com/world")
    assert reuters_score.authority_level == 4
    assert 0.75 <= reuters_score.credibility_score <= 0.90

    # Level 6: Low authority
    reddit_score = analyze_source_credibility("https://www.reddit.com/r/facts")
    assert reddit_score.authority_level == 6
    assert reddit_score.credibility_score <= 0.40


def test_belief_vector_probabilities():
    bv = BeliefVector(supported=0.6, refuted=0.2, unknown=0.2)
    assert np.isclose(bv.supported + bv.refuted + bv.unknown, 1.0)
    assert 0.0 <= bv.supported <= 1.0
    assert 0.0 <= bv.refuted <= 1.0
    assert 0.0 <= bv.unknown <= 1.0


def test_autonomous_routing_logic():
    state = initial_veritas_state("Test claim")
    state["current_claim"] = AtomicClaim(
        claim_id="C1", claim_text="Test", claim_type="factual", entities=[],
        temporal_context=None, verification_questions=[], source_span="Test"
    )

    # 1. Zero evidence -> routes to autonomous_research
    state["evidence_pool"] = []
    state["research_rounds"] = 0
    route = route_after_consensus(state)
    assert route == "autonomous_research"

    # 2. Ample evidence, high disagreement -> routes to debate loop
    state["evidence_pool"] = [
        EvidenceItem(
            evidence_id=f"E{i}", claim_id="C1", title="T", url=f"http://test{i}.com",
            snippet="S", source_domain="test.com"
        )
        for i in range(5)
    ]
    state["research_rounds"] = 2
    state["belief_vectors"] = {
        "advocate": BeliefVector(supported=0.9, refuted=0.05, unknown=0.05),
        "skeptic": BeliefVector(supported=0.1, refuted=0.85, unknown=0.05),
    }
    state["current_round"] = 0
    state["max_rounds"] = 3
    route2 = route_after_consensus(state)
    assert route2 == ["advocate", "skeptic", "neutral"]

    # 3. Consensus reached -> routes to final_verifier
    state["belief_vectors"] = {
        "advocate": BeliefVector(supported=0.85, refuted=0.08, unknown=0.07),
        "skeptic": BeliefVector(supported=0.82, refuted=0.09, unknown=0.09),
    }
    state["consensus_converged"] = True
    route3 = route_after_consensus(state)
    assert route3 == "final_verifier"


def test_full_veritas_audit_execution():
    result = run_audit(
        text="The Eiffel Tower is 330 meters tall and was built in 1889.",
        mode="veritas",
        max_rounds=2,
    )
    assert result.session_id is not None
    assert len(result.claims) >= 1
    assert len(result.verdicts) >= 1
    assert result.aggregate_verdict in [VerdictType.SUPPORTED, VerdictType.REFUTED, VerdictType.NOT_ENOUGH_INFORMATION]
    assert 0.0 <= result.aggregate_confidence <= 1.0
    assert len(result.audit_trail) > 0
    assert result.execution_time_seconds >= 0.0


def test_not_enough_information_verdict():
    result = run_audit(
        text="Project Titan discovered sentient extraterrestrial signals in deep space in March 2026.",
        mode="veritas",
        max_rounds=2,
    )
    assert result.aggregate_verdict == VerdictType.NOT_ENOUGH_INFORMATION
    assert result.verdicts[0].why_not_other_verdicts is not None
