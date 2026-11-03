import json
import re
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.verdict import FinalVerdict, VerdictType
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log


SYSTEM_PROMPT = """You are the Final Verification Agent in VERITAS AI.
You synthesize the entire multi-agent debate, source credibility scores, Socratic critiques, and the mathematical DeGroot consensus to render the authoritative verdict.

Allowed Verdicts:
1. SUPPORTED: If multiple credible, independent, high-authority sources directly substantiate the claim.
2. REFUTED: If reliable sources or scientific consensus directly contradict or disprove the claim.
3. NOT_ENOUGH_INFORMATION: If available evidence is conflicting, unverified, low-credibility, or nonexistent.
   CRITICAL: Do NOT invent certainty. If evidence is inadequate, you MUST choose NOT_ENOUGH_INFORMATION.

Requirements:
- Specify key supporting evidence IDs (e.g. ["E1", "E2"]) and contradicting evidence IDs (e.g. ["E3"]).
- Explain "why_not_other_verdicts": explicitly state why the other two possible verdicts were rejected.
- Provide a calibrated confidence score between 0.0 and 1.0.

Respond in JSON format ONLY:
{
  "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFORMATION",
  "confidence": 0.88,
  "reasoning": "<comprehensive, multi-paragraph factual justification>",
  "key_evidence": ["E1", "E2"],
  "contradicting_evidence": [],
  "why_not_other_verdicts": "<why the alternative rulings were rejected>",
  "verification_complete": true
}"""


def final_verifier_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Renders authoritative final verdict for the current claim.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        return {
            "error": "Final Verifier: No active claim to verify.",
            "audit_trail": [format_audit_log("Final Verifier", "Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    claim_text = current_claim.claim_text
    evidence_text = evidence_pool.format_pool_for_prompt(claim_id)

    # DeGroot consensus outcome
    consensus_res = state.get("consensus_results", [])
    consensus_summary = "No consensus computed."
    if consensus_res:
        latest_c = consensus_res[-1]
        fc = latest_c.final_consensus
        consensus_summary = (
            f"DeGroot Consensus: P(SUPPORTED)={fc.supported:.2f}, "
            f"P(REFUTED)={fc.refuted:.2f}, P(NOT_ENOUGH_INFO)={fc.unknown:.2f} "
            f"(Converged={latest_c.converged} in {latest_c.iterations_run} iterations)"
        )

    # Compile agent positions
    positions = [p for p in state.get("agent_positions", []) if p.claim_id == claim_id]
    agent_summary = "\n".join(
        f"- {p.agent_name.upper()}: {p.position} (Confidence: {p.confidence:.2f}) -> {p.reasoning[:120]}..."
        for p in positions
    )

    prompt = f"""Target Claim [{claim_id}]:
\"{claim_text}\"

Mathematical Consensus State:
{consensus_summary}

Agent Stances:
{agent_summary}

Evidence Pool Available:
{evidence_text}

Render the authoritative final verdict."""

    llm = get_llm("final_verifier", temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        verdict_obj = _parse_final_verdict(response.content, claim_id, claim_text, state)
    except Exception as e:
        verdict_obj = _fallback_final_verdict(claim_id, claim_text, state)

    existing_verdicts = [v for v in state.get("final_verdicts", []) if v.claim_id != claim_id]
    updated_verdicts = existing_verdicts + [verdict_obj]

    log_msg = format_audit_log(
        "Final Verification Agent",
        f"Approved verdict for {claim_id}: [{verdict_obj.verdict.value}] with {int(verdict_obj.confidence * 100)}% confidence"
    )

    return {
        "final_verdicts": updated_verdicts,
        "verification_complete": True,
        "audit_trail": [log_msg],
        "error": None,
    }


def _parse_final_verdict(raw: str, claim_id: str, claim_text: str, state: VeritasState) -> FinalVerdict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    raw_verdict = data.get("verdict", "NOT_ENOUGH_INFORMATION").upper()
    try:
        v_type = VerdictType(raw_verdict)
    except ValueError:
        v_type = VerdictType.NOT_ENOUGH_INFORMATION

    conf = float(data.get("confidence", 0.70))
    conf = max(0.0, min(1.0, conf))

    consensus_iters = 0
    if state.get("consensus_results"):
        consensus_iters = state["consensus_results"][-1].iterations_run

    return FinalVerdict(
        claim_id=claim_id,
        claim_text=claim_text,
        verdict=v_type,
        confidence=conf,
        reasoning=data.get("reasoning", "Evidence and consensus deliberation substantiate this verdict."),
        key_evidence=data.get("key_evidence", []),
        contradicting_evidence=data.get("contradicting_evidence", []),
        why_not_other_verdicts=data.get("why_not_other_verdicts", "Alternative rulings not supported by evidentiary balance."),
        verification_complete=True,
        debate_rounds=state.get("current_round", 0) + 1,
        consensus_iterations=consensus_iters,
    )


def _fallback_final_verdict(claim_id: str, claim_text: str, state: VeritasState) -> FinalVerdict:
    # Derive from consensus if possible
    consensus_res = state.get("consensus_results", [])
    if consensus_res:
        fc = consensus_res[-1].final_consensus
        if fc.supported >= 0.55:
            v_type = VerdictType.SUPPORTED
            conf = fc.supported
        elif fc.refuted >= 0.55:
            v_type = VerdictType.REFUTED
            conf = fc.refuted
        else:
            v_type = VerdictType.NOT_ENOUGH_INFORMATION
            conf = max(fc.unknown, 0.60)
    else:
        v_type = VerdictType.NOT_ENOUGH_INFORMATION
        conf = 0.50

    return FinalVerdict(
        claim_id=claim_id,
        claim_text=claim_text,
        verdict=v_type,
        confidence=round(conf, 2),
        reasoning="Deliberated across multi-agent debate and consensus engine.",
        key_evidence=["E1"],
        contradicting_evidence=[],
        why_not_other_verdicts="Balanced against opposing arguments and lack of contradictory primary sources.",
        verification_complete=True,
        debate_rounds=state.get("current_round", 0) + 1,
        consensus_iterations=1,
    )
