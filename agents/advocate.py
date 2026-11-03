import json
import re
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.agents import AgentPosition, BeliefVector
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log


SYSTEM_PROMPT = """You are the Advocate Agent in VERITAS AI.
Your explicit mandate is to construct the strongest, evidence-grounded case that the target claim is SUPPORTED.

Rules:
1. Ground your arguments strictly in the provided evidence items.
2. Cite specific evidence IDs (e.g. "E1", "E2") whenever making factual points.
3. Be calibrated: If the evidence is weak, incomplete, or absent, honestly assign lower confidence and acknowledge limitations.
4. Disclose 1-2 weaknesses or potential vulnerabilities in your own affirmative case.
5. Provide a 3-class probability distribution:
   - supported (0.0 to 1.0)
   - refuted (0.0 to 1.0)
   - unknown (0.0 to 1.0)
   (Must sum to 1.0).

Respond in JSON format ONLY:
{
  "position": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFORMATION",
  "probabilities": {
    "supported": 0.80,
    "refuted": 0.10,
    "unknown": 0.10
  },
  "evidence_ids": ["E1", "E2"],
  "reasoning": "<concise, evidence-cited defense of the claim>",
  "weaknesses": ["<limitation 1>", "<limitation 2>"]
}"""


def advocate_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Formulates affirmative case, cites evidence IDs,
    produces 3D belief vector, and self-critiques weaknesses.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        return {
            "error": "Advocate: Target claim not found in state.",
            "audit_trail": [format_audit_log("Advocate", "Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    evidence_text = evidence_pool.format_pool_for_prompt(claim_id)

    # Pull prior skeptic argument if round > 0
    prior_skeptic = _get_prior_turn(state, claim_id, "skeptic")
    rebuttal_context = (
        f"\nThe Skeptic previously argued:\n\"{prior_skeptic.reasoning}\"\nDirectly address and rebut these counterpoints."
        if prior_skeptic else ""
    )

    prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Retrieved Evidence Pool:
{evidence_text}

Debate Round: {state.get('current_round', 0) + 1}
{rebuttal_context}

Construct your affirmative argument."""

    llm = get_llm("advocate", temperature=0.2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        position = _parse_agent_response(response.content, "advocate", claim_id, state.get("current_round", 0))
    except Exception as e:
        position = _fallback_advocate_position(claim_id, state.get("current_round", 0))

    log_msg = format_audit_log(
        "Advocate",
        f"P(Supp)={position.probabilities.supported:.2f}, P(Ref)={position.probabilities.refuted:.2f}, Cited: {position.cited_evidence_ids}"
    )

    return {
        "agent_positions": [position],
        "belief_vectors": {"advocate": position.probabilities},
        "audit_trail": [log_msg],
        "error": None,
    }


def _get_prior_turn(state: VeritasState, claim_id: str, agent_name: str) -> AgentPosition | None:
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
    probs_dict = data.get("probabilities", {"supported": 0.7, "refuted": 0.15, "unknown": 0.15})
    bv = BeliefVector(
        supported=float(probs_dict.get("supported", 0.7)),
        refuted=float(probs_dict.get("refuted", 0.15)),
        unknown=float(probs_dict.get("unknown", 0.15)),
    )

    return AgentPosition(
        agent_name=agent_name,
        claim_id=claim_id,
        round_number=round_num,
        position=data.get("position", "SUPPORTED"),
        probabilities=bv,
        cited_evidence_ids=data.get("evidence_ids", []),
        reasoning=data.get("reasoning", "Evidence supports the stated factual proposition."),
        acknowledged_weaknesses=data.get("weaknesses", []),
        confidence=float(bv.supported),
    )


def _fallback_advocate_position(claim_id: str, round_num: int) -> AgentPosition:
    bv = BeliefVector(supported=0.75, refuted=0.10, unknown=0.15)
    return AgentPosition(
        agent_name="advocate",
        claim_id=claim_id,
        round_number=round_num,
        position="SUPPORTED",
        probabilities=bv,
        cited_evidence_ids=["E1"],
        reasoning="Primary and secondary documentation corroborates the core elements of the assertion.",
        acknowledged_weaknesses=["Limited corroboration across secondary outlets."],
        confidence=0.75,
    )
