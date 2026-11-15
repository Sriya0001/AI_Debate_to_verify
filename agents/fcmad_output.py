import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.verdict import FinalVerdict, VerdictType
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log
from config import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Output Agent in the FC-MAD verification system.
The mathematical consensus engine has ALREADY finalized the verdict and probabilities:
- Verdict: {verdict}
- Distribution: P(Supp)={p_supp:.2f}, P(Ref)={p_ref:.2f}, P(Unk)={p_unk:.2f}
- Confidence: {confidence:.1%} (Abstain={abstain})

YOUR SOLE DUTY is to write a clear, factual, and rigorous narrative explanation:
1. Explain why the evidence and debate deliberation corroborate, refute, or leave this claim uncertain.
2. Directly reference key cited evidence IDs.
3. State why alternative verdicts were rejected.

CRITICAL INSTRUCTION:
You MUST NOT change the verdict or the probability distribution. Write the explanation only."""


def fcmad_output_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Writes narrative explanation for the mathematically finalized consensus verdict.
    Never modifies the verdict label, probabilities, or confidence.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        err_msg = "Output Agent: Target claim not found in state."
        return {
            "error": err_msg,
            "recorded_failures": [{"agent": "output_agent", "error": err_msg}],
            "audit_trail": [format_audit_log("Output Agent", f"Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    final_dist = state.get("final_distribution") or [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    raw_conf = state.get("raw_confidence") or float(max(final_dist))
    cal_conf = state.get("calibrated_confidence") or raw_conf
    abstain_flag = state.get("abstain", False)

    # Effective confidence to report
    reported_conf = cal_conf if config.calibration_enabled else raw_conf

    # Determine verdict label
    import numpy as np
    classes = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
    winning_idx = int(np.argmax(final_dist))
    label_str = classes[winning_idx]
    verdict_enum = getattr(VerdictType, label_str, VerdictType.NOT_ENOUGH_INFORMATION)

    # Gather cited evidence
    positions = state.get("agent_positions", [])
    claim_positions = [p for p in positions if p.claim_id == claim_id]
    all_cited = list(dict.fromkeys([
        cid for p in claim_positions for cid in p.cited_evidence_ids
    ]))

    sys_prompt = SYSTEM_PROMPT.format(
        verdict=label_str if not abstain_flag else f"{label_str} (ABSTAINED: UNCERTAIN)",
        p_supp=final_dist[0],
        p_ref=final_dist[1],
        p_unk=final_dist[2],
        confidence=reported_conf,
        abstain=abstain_flag,
    )

    evidence_text = evidence_pool.format_pool_for_prompt(claim_id)
    summary_text = state.get("fcmad_debate_summary", "Deliberation completed across debate rounds.")

    user_prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Mathematical Consensus Distribution:
- P(Supported): {final_dist[0]:.4f}
- P(Refuted):   {final_dist[1]:.4f}
- P(Unknown):   {final_dist[2]:.4f}
Final Verdict: {label_str} (Confidence: {reported_conf:.1%})

Debate Summary:
{summary_text}

Evidence Pool:
{evidence_text}

Cited Evidence IDs: {all_cited}

Write a concise narrative explanation justifying this verdict."""

    llm = get_llm("output_agent", model_name=getattr(config, "output_agent_model", None), temperature=0.0)
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt),
    ]

    explanation = None
    for attempt in range(config.max_retries):
        try:
            resp = llm.invoke(messages)
            explanation = resp.content.strip()
            break
        except Exception as e:
            logger.warning("Output Agent attempt %d/%d failed: %s", attempt + 1, config.max_retries, e)

    if not explanation:
        explanation = (
            f"Consensus deliberation concluded with verdict {label_str} (Confidence: {reported_conf:.1%}). "
            f"Credibility-weighted belief distribution: P(S)={final_dist[0]:.2f}, P(R)={final_dist[1]:.2f}, P(N)={final_dist[2]:.2f}."
        )

    rounds_used = state.get("rounds_used") or max(1, state.get("current_round", 0))
    verdict_obj = FinalVerdict(
        claim_id=claim_id,
        claim_text=current_claim.claim_text,
        verdict=verdict_enum,
        confidence=float(round(reported_conf, 4)),
        reasoning=explanation,
        key_evidence=all_cited if label_str == "SUPPORTED" else [],
        contradicting_evidence=all_cited if label_str == "REFUTED" else [],
        why_not_other_verdicts=f"Alternative verdicts rejected based on consensus probabilities P(S)={final_dist[0]:.2f}, P(R)={final_dist[1]:.2f}, P(N)={final_dist[2]:.2f}.",
        verification_complete=True,
        debate_rounds=rounds_used,
        consensus_iterations=len(state.get("disagreement_history", [])),
        final_distribution=[float(round(v, 4)) for v in final_dist],
        raw_confidence=float(round(raw_conf, 4)),
        calibrated_confidence=float(round(cal_conf, 4)),
        abstain=abstain_flag,
        recorded_failures=state.get("recorded_failures", []),
        credibilities=state.get("credibilities", []),
        disagreement_history=state.get("disagreement_history", []),
        rounds_used=rounds_used,
    )

    log_msg = format_audit_log(
        "Output Agent (FC-MAD)",
        f"Finalized verdict: {label_str} ({reported_conf:.1%}) | Abstain={abstain_flag} | Explanation: {explanation[:80]}..."
    )

    return {
        "final_verdicts": [verdict_obj],
        "aggregate_verdict": verdict_enum if not abstain_flag else VerdictType.NOT_ENOUGH_INFORMATION,
        "aggregate_confidence": float(round(reported_conf, 4)),
        "audit_summary": f"FC-MAD Verdict: {label_str} (Confidence: {reported_conf:.1%}, Abstain={abstain_flag}). {explanation}",
        "verification_complete": True,
        "audit_trail": [log_msg],
        "error": None,
    }
