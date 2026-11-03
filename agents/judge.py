import json
import re
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.agents import JudgeEvaluation, AgentPosition
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log


SYSTEM_PROMPT = """You are the Judge Agent in VERITAS AI.
Your purpose is to evaluate the debate performance, evidence reliance, source credibility grounding, and logical consistency of the reasoning agents:
1. Advocate
2. Skeptic
3. Neutral Analyst
4. Socratic Critic

Assign an objective, normalized trust score to each agent.
Requirements:
1. All weights must be non-negative and sum to 1.0.
2. Agents that cite high-credibility primary/regulatory/academic sources earn higher trust.
3. Agents making unsupported assertions, dogmatic claims, or ignoring counter-evidence receive lower trust.
4. You must provide an explicit justification explaining why each trust weight was assigned.

Schema:
Respond in JSON format ONLY:
{
  "agent_trust_weights": {
    "advocate": 0.30,
    "skeptic": 0.35,
    "neutral": 0.25,
    "critic": 0.10
  },
  "argument_quality_scores": {
    "advocate": 0.85,
    "skeptic": 0.88,
    "neutral": 0.80,
    "critic": 0.75
  },
  "evidence_reliance_scores": {
    "advocate": 0.90,
    "skeptic": 0.85,
    "neutral": 0.80,
    "critic": 0.70
  },
  "rationale": "<comprehensive explanation of trust weight allocation>"
}"""


def judge_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Evaluates debate argumentation, quality of evidence cited,
    and dynamically computes normalized trust weights for the DeGroot model.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        return {
            "error": "Judge: Target claim not found in state.",
            "audit_trail": [format_audit_log("Judge", "Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    evidence_text = evidence_pool.format_pool_for_prompt(claim_id)

    # Gather latest positions for each agent
    positions: List[AgentPosition] = state.get("agent_positions", [])
    claim_positions = [p for p in positions if p.claim_id == claim_id]

    turns_summary = []
    for p in claim_positions:
        turns_summary.append(
            f"Agent: {p.agent_name.upper()}\n"
            f"Stance: {p.position} | Belief Vector: [Supp={p.probabilities.supported:.2f}, Ref={p.probabilities.refuted:.2f}, Unk={p.probabilities.unknown:.2f}]\n"
            f"Cited Evidence IDs: {p.cited_evidence_ids}\n"
            f"Argument: {p.reasoning}\n"
            f"Acknowledged Weaknesses: {p.acknowledged_weaknesses}\n"
        )
    transcript_text = "\n---\n".join(turns_summary) or "No arguments logged."

    # Format source credibility scores
    sources_summary = "\n".join(
        f"- {domain}: Credibility {s.credibility_score:.2f} ({s.authority_name}) | Primary={s.is_primary}"
        for domain, s in state.get("source_scores", {}).items()
    ) or "No source scores evaluated."

    prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Source Credibility Hierarchy & Scores:
{sources_summary}

Evidence Pool Available:
{evidence_text}

Debate Performance Transcript:
{transcript_text}

Evaluate argument quality, source reliance, and assign normalized trust weights."""

    llm = get_llm("judge", temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        evaluation = _parse_judge_response(response.content, claim_id, state.get("current_round", 0))
    except Exception as e:
        evaluation = _fallback_judge_evaluation(claim_id, state.get("current_round", 0))

    tw_str = ", ".join(f"{k}: {v:.2f}" for k, v in evaluation.agent_trust_weights.items())
    log_msg = format_audit_log(
        "Judge",
        f"Assigned trust weights: [{tw_str}] | Rationale: {evaluation.rationale[:80]}..."
    )

    return {
        "judge_evaluations": [evaluation],
        "trust_weights": evaluation.agent_trust_weights,
        "audit_trail": [log_msg],
        "error": None,
    }


def _parse_judge_response(raw: str, claim_id: str, round_num: int) -> JudgeEvaluation:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    weights = data.get("agent_trust_weights", {
        "advocate": 0.30,
        "skeptic": 0.35,
        "neutral": 0.25,
        "critic": 0.10,
    })

    return JudgeEvaluation(
        claim_id=claim_id,
        round_number=round_num,
        agent_trust_weights=weights,
        argument_quality_scores=data.get("argument_quality_scores", {}),
        evidence_reliance_scores=data.get("evidence_reliance_scores", {}),
        rationale=data.get("rationale", "Trust weighted by evidence grounding and argument quality."),
    )


def _fallback_judge_evaluation(claim_id: str, round_num: int) -> JudgeEvaluation:
    return JudgeEvaluation(
        claim_id=claim_id,
        round_number=round_num,
        agent_trust_weights={
            "advocate": 0.30,
            "skeptic": 0.30,
            "neutral": 0.25,
            "critic": 0.15,
        },
        argument_quality_scores={"advocate": 0.8, "skeptic": 0.8, "neutral": 0.8, "critic": 0.75},
        evidence_reliance_scores={"advocate": 0.85, "skeptic": 0.85, "neutral": 0.8, "critic": 0.7},
        rationale="Balanced initial trust distribution with slight preference for primary debaters.",
    )
