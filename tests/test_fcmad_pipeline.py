import pytest
from runner import run_audit
from models.verdict import VerdictType
from config import config


# ── FCMAD Integration Test 1: Debaters agree (stops early) ─────────────────────

def test_fcmad_debaters_agree_stops_early():
    """
    Scenario 1: Debaters agree on Round 0 (disagreement E <= epsilon).
    The FCMAD stop router must stop early on round 0/1, bypassing extra debate rounds,
    and proceed directly to pooling and the output agent.
    """
    result = run_audit(
        text="The Eiffel Tower is located in Paris and was completed in 1889.",
        mode="fcmad",
        max_rounds=4,
    )

    assert result.session_id is not None
    assert result.mode == "fcmad"
    assert len(result.claims) >= 1
    assert len(result.verdicts) >= 1

    v = result.verdicts[0]
    assert v.verdict == VerdictType.SUPPORTED
    assert v.confidence >= 0.70
    assert v.final_distribution is not None
    assert len(v.final_distribution) == 3
    assert abs(sum(v.final_distribution) - 1.0) < 1e-4

    # Stopped early on Round 1 (at or before 2 rounds)
    assert v.debate_rounds <= 2
    assert len(v.disagreement_history) >= 1
    # Latest disagreement is below epsilon
    assert v.disagreement_history[-1] <= config.epsilon

    # Output agent explanation is present
    assert len(v.reasoning) > 10
    assert result.aggregate_verdict == VerdictType.SUPPORTED


# ── FCMAD Integration Test 2: Debaters disagree then converge ──────────────────

def test_fcmad_debaters_disagree_then_converge():
    """
    Scenario 2: Debaters initially disagree (E > epsilon), triggering the Summarizer
    and revision loop. In Round 1, debaters re-reason from the summary and converge,
    stopping before hitting max rounds.
    """
    result = run_audit(
        text="The historical committee convened to converge on a unified protocol.",
        mode="fcmad",
        max_rounds=4,
    )

    assert result.session_id is not None
    assert result.mode == "fcmad"
    assert len(result.verdicts) >= 1

    v = result.verdicts[0]
    # Disagreement history shows multiple rounds
    assert len(v.disagreement_history) >= 2
    # Initial round had high disagreement
    assert v.disagreement_history[0] > config.epsilon
    # Final round converged within epsilon
    assert v.disagreement_history[-1] <= config.epsilon

    # Finished before hitting max rounds (e.g. 2 rounds used out of 4)
    assert v.debate_rounds < 4
    assert v.final_distribution is not None
    assert result.aggregate_verdict in [VerdictType.SUPPORTED, VerdictType.REFUTED, VerdictType.NOT_ENOUGH_INFORMATION]


# ── FCMAD Integration Test 3: Debaters persistently disagree (hits max rounds) ──

def test_fcmad_debaters_disagree_hits_max_rounds():
    """
    Scenario 3: Debaters persistently disagree across all rounds (E > epsilon).
    The FCMAD stop router must continue iterating until current_round reaches max_rounds,
    stopping on the max rounds rule.
    """
    max_r = 3
    result = run_audit(
        text="The ancient border boundary remains under persistent dispute.",
        mode="fcmad",
        max_rounds=max_r,
    )

    assert result.session_id is not None
    assert result.mode == "fcmad"
    assert len(result.verdicts) >= 1

    v = result.verdicts[0]
    # Reached the exact maximum rounds cutoff
    assert v.debate_rounds == max_r
    # Disagreement remained above epsilon
    assert v.disagreement_history[-1] > config.epsilon
    assert v.final_distribution is not None
    assert abs(sum(v.final_distribution) - 1.0) < 1e-4
    assert 0.0 <= v.confidence <= 1.0


# ── FCMAD Integration Test 4: Provided evidence skips external retrieval ───────

def test_fcmad_provided_gold_evidence_skips_search():
    """
    Verifies that passing gold evidence directly to run_audit populates the
    evidence pool and skips external web search.
    """
    gold_evidence = [
        {
            "evidence_id": "GOLD_1",
            "title": "Official UNESCO Registry",
            "snippet": "The landmark was constructed between 1887 and 1889 as the entrance arch to the 1889 World's Fair.",
            "source_domain": "unesco.org",
            "supports_claim": True,
            "relevance_score": 1.0,
        },
        {
            "evidence_id": "GOLD_2",
            "title": "Architectural Archives",
            "snippet": "Height measured exactly at 330 meters to top tip.",
            "source_domain": "archives.fr",
            "supports_claim": True,
            "relevance_score": 0.95,
        }
    ]

    result = run_audit(
        text="The Eiffel Tower is 330 meters tall.",
        mode="fcmad",
        max_rounds=2,
        evidence=gold_evidence,
    )

    assert result.mode == "fcmad"
    assert len(result.verdicts) >= 1
    # Check that external searches were 0 because provided evidence was used
    assert result.total_searches == 0

    # Verify audit trail records skipped external retrieval
    audit_text = " ".join(result.audit_trail)
    assert "provided gold evidence" in audit_text.lower() or "skipped external retrieval" in audit_text.lower()


# ── FCMAD Integration Test 5: Output Agent does not modify label or distribution ──

def test_fcmad_output_agent_preserves_label_and_distribution():
    """
    Verifies that the Output Agent only generates the narrative explanation
    and never modifies the verdict or probability distribution.
    """
    result = run_audit(
        text="Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        mode="fcmad",
        max_rounds=2,
    )

    v = result.verdicts[0]
    dist = v.final_distribution
    assert dist is not None

    import numpy as np
    classes = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
    expected_winning_idx = int(np.argmax(dist))
    expected_label = classes[expected_winning_idx]

    # The verdict enum must match the mathematical argmax
    assert v.verdict.value == expected_label
    # Raw confidence must equal max probability in distribution
    assert abs(v.raw_confidence - float(dist[expected_winning_idx])) < 1e-4


# ── FCMAD Integration Test 6: Heterogeneous Cross-Model Debater Assignment ─────

def test_fcmad_heterogeneous_multi_model_debater_assignment():
    """
    Verifies that distinct models can be assigned across debater agents,
    enabling heterogeneous cross-model deliberation.
    """
    custom_models = [
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ]
    result = run_audit(
        text="The Eiffel Tower is 330 meters tall including antennas and was completed in 1889.",
        mode="fcmad",
        max_rounds=2,
        debater_models=custom_models,
    )

    assert result.session_id is not None
    assert len(result.verdicts) >= 1
    audit_text = " ".join(result.audit_trail)
    
    # Verify that the audit trail explicitly recorded the distinct model names
    assert "llama-3.3-70b-versatile" in audit_text
    assert "gemma2-9b-it" in audit_text
    assert "mixtral-8x7b-32768" in audit_text
