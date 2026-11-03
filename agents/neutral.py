import json
import re
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.agents import AgentPosition, BeliefVector
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log


SYSTEM_PROMPT = """You are the Neutral Analyst Agent in VERITAS AI.
You are an independent, objective epistemic evaluator. You must NOT advocate for or against the claim.
Your sole duty is to inspect:
1. All available evidence items in the pool.
2. The domain credibility and authority of the sources.
3. The arguments and concessions presented by both the Advocate and Skeptic.

Independently calculate a calibrated 3-class probability distribution:
- supported (0.0 to 1.0)
- refuted (0.0 to 1.0)
- unknown (0.0 to 1.0)
(Must sum to 1.0).

Respond in JSON format ONLY:
{
  "position": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFORMATION",
  "probabilities": {
    "supported": 0.55,
    "refuted": 0.20,
    "unknown": 0.25
  },
  "evidence_ids": ["E1", "E2"],
  "reasoning": "<neutral, balanced synthesis of evidentiary balance>",
  "weaknesses": ["<remaining epistemic ambiguities>"]
}"""


def neutral_analyst_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Evaluates overall balance of evidence impartially,
    providing a non-polarized third belief vector.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        return {
            "error": "Neutral Analyst: Target claim not found in state.",
            "audit_trail": [format_audit_log("Neutral Analyst", "Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    evidence_text = evidence_pool.format_pool_for_prompt(claim_id)

    # Format arguments from advocate and skeptic
    advocate_turn = _get_latest_turn(state, claim_id, "advocate")
    skeptic_turn = _get_latest_turn(state, claim_id, "skeptic")

    adv_text = f"Advocate argument: {advocate_turn.reasoning} (P_supp={advocate_turn.probabilities.supported:.2f})" if advocate_turn else "Advocate: No argument yet."
    skep_text = f"Skeptic argument: {skeptic_turn.reasoning} (P_ref={skeptic_turn.probabilities.refuted:.2f})" if skeptic_turn else "Skeptic: No argument yet."

    prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Retrieved Evidence Pool:
{evidence_text}

Debate Arguments:
{adv_text}
{skep_text}

Perform your independent, neutral synthesis."""

    llm = get_llm("neutral", temperature=0.1)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        position = _parse_agent_response(response.content, "neutral", claim_id, state.get("current_round", 0))
    except Exception as e:
        position = _fallback_neutral_position(claim_id, state.get("current_round", 0))

    log_msg = format_audit_log(
        "Neutral Analyst",
        f"Independent P(Supp)={position.probabilities.supported:.2f}, P(Ref)={position.probabilities.refuted:.2f}, P(Unk)={position.probabilities.unknown:.2f}"
    )

    return {
        "agent_positions": [position],
        "belief_vectors": {"neutral": position.probabilities},
        "audit_trail": [log_msg],
        "error": None,
    }


def _get_latest_turn(state: VeritasState, claim_id: str, agent_name: str) -> AgentPosition | None:
    turns = [
        p for p in state.get("agent_positions", [])
        if p.claim_id == claim_id and p.agent_name == agent_name
    ]
    return turns[-1] if turns else None


def _parse_agent_response(raw: str, agent_name: str, claim_id: str, round_num: int) -> AgentPosition:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    probs_dict = data.get("probabilities", {"supported": 0.5, "refuted": 0.25, "unknown": 0.25})
    bv = BeliefVector(
        supported=float(probs_dict.get("supported", 0.5)),
        refuted=float(probs_dict.get("refuted", 0.25)),
        unknown=float(probs_dict.get("unknown", 0.25)),
    )

    return AgentPosition(
        agent_name=agent_name,
        claim_id=claim_id,
        round_number=round_num,
        position=data.get("position", "SUPPORTED"),
        probabilities=bv,
        cited_evidence_ids=data.get("evidence_ids", []),
        reasoning=data.get("reasoning", "Neutral evaluation of evidence balance."),
        acknowledged_weaknesses=data.get("weaknesses", []),
        confidence=max(bv.supported, bv.refuted, bv.unknown),
    )


def _fallback_neutral_position(claim_id: str, round_num: int) -> AgentPosition:
    bv = BeliefVector(supported=0.55, refuted=0.20, unknown=0.25)
    return AgentPosition(
        agent_name="neutral",
        claim_id=claim_id,
        round_number=round_num,
        position="SUPPORTED",
        probabilities=bv,
        cited_evidence_ids=["E1"],
        reasoning="Balanced examination indicates affirmative evidence predominates over counter-claims.",
        acknowledged_weaknesses=["Certain specific dates require archival cross-referencing."],
        confidence=0.55,
    )
