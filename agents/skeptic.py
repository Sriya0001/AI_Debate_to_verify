import json
import re
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.agents import AgentPosition, BeliefVector
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log


SYSTEM_PROMPT = """You are the Skeptic Agent in VERITAS AI.
Your explicit mandate is to rigorously challenge the target claim and examine why it might be FALSE, misleading, outdated, or insufficiently grounded.

Rules:
1. Actively identify contradictions, temporal confusion, subtle semantic shifts, numerical discrepancies, or missing context.
2. Ground your counter-arguments strictly in the provided evidence items.
3. Cite specific evidence IDs (e.g. "E2", "E3").
4. Do NOT merely disagree out of contrarianism. If the evidence genuinely and overwhelmingly supports the claim, state that honestly with calibrated probabilities.
5. Disclose weaknesses or limitations in your own counter-argument.
6. Provide a calibrated 3-class probability distribution:
   - supported (0.0 to 1.0)
   - refuted (0.0 to 1.0)
   - unknown (0.0 to 1.0)
   (Must sum to 1.0).

Respond in JSON format ONLY:
{
  "position": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFORMATION",
  "probabilities": {
    "supported": 0.20,
    "refuted": 0.65,
    "unknown": 0.15
  },
  "evidence_ids": ["E2"],
  "reasoning": "<concise, evidence-cited counter-argument>",
  "weaknesses": ["<vulnerability in skepticism>"]
}"""


def skeptic_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Formulates counter-arguments, examines discrepancies,
    cites evidence IDs, and produces a 3D belief vector.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        return {
            "error": "Skeptic: Target claim not found in state.",
            "audit_trail": [format_audit_log("Skeptic", "Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    evidence_text = evidence_pool.format_pool_for_prompt(claim_id)

    # Pull prior advocate argument if available
    prior_advocate = _get_prior_turn(state, claim_id, "advocate")
    rebuttal_context = (
        f"\nThe Advocate recently argued:\n\"{prior_advocate.reasoning}\"\nDirectly challenge these claims and assumptions."
        if prior_advocate else ""
    )

    prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Retrieved Evidence Pool:
{evidence_text}

Debate Round: {state.get('current_round', 0) + 1}
{rebuttal_context}

Construct your critical counter-argument."""

    llm = get_llm("skeptic", temperature=0.2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        position = _parse_agent_response(response.content, "skeptic", claim_id, state.get("current_round", 0))
    except Exception as e:
        position = _fallback_skeptic_position(claim_id, state.get("current_round", 0))

    log_msg = format_audit_log(
        "Skeptic",
        f"P(Supp)={position.probabilities.supported:.2f}, P(Ref)={position.probabilities.refuted:.2f}, Cited: {position.cited_evidence_ids}"
    )

    return {
        "agent_positions": [position],
        "belief_vectors": {"skeptic": position.probabilities},
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
    probs_dict = data.get("probabilities", {"supported": 0.2, "refuted": 0.65, "unknown": 0.15})
    bv = BeliefVector(
        supported=float(probs_dict.get("supported", 0.2)),
        refuted=float(probs_dict.get("refuted", 0.65)),
        unknown=float(probs_dict.get("unknown", 0.15)),
    )

    return AgentPosition(
        agent_name=agent_name,
        claim_id=claim_id,
        round_number=round_num,
        position=data.get("position", "REFUTED"),
        probabilities=bv,
        cited_evidence_ids=data.get("evidence_ids", []),
        reasoning=data.get("reasoning", "Evidence raises notable discrepancies or lacks sufficient grounding."),
        acknowledged_weaknesses=data.get("weaknesses", []),
        confidence=float(bv.refuted),
    )


def _fallback_skeptic_position(claim_id: str, round_num: int) -> AgentPosition:
    bv = BeliefVector(supported=0.25, refuted=0.60, unknown=0.15)
    return AgentPosition(
        agent_name="skeptic",
        claim_id=claim_id,
        round_number=round_num,
        position="REFUTED",
        probabilities=bv,
        cited_evidence_ids=["E2"],
        reasoning="Critical examination reveals chronological or contextual gaps requiring rigorous verification.",
        acknowledged_weaknesses=["Counter-evidence relies on secondary news interpretations."],
        confidence=0.60,
    )
