import json
import re
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.agents import AgentPosition, BeliefVector, SocraticChallenge
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log


SYSTEM_PROMPT = """You are the Socratic Critic in VERITAS AI.
Your purpose is epistemological stress-testing. You audit the reasoning of the Advocate, Skeptic, and Neutral Analyst.

Probe for:
1. Entity Identity Drift: Are all agents discussing the exact same entity, specification, or event?
2. Temporal Validity: Could the cited evidence be superseded by newer developments?
3. Correlation vs. Causation: Is an assumed causal link actually proven by evidence?
4. Hidden Assumptions: What unspoken premise is the argument dependent upon?
5. Evidence Gaps: Does a specific claim lack direct corroboration? If so, formulate a targeted research query.

Schema:
Respond in JSON format ONLY:
{
  "challenges": [
    {
      "challenge_id": "SC1",
      "target_agent": "advocate" | "skeptic" | "all",
      "question": "<probing Socratic question>",
      "challenge_type": "temporal_drift" | "entity_mismatch" | "correlation_causation" | "unsupported_assumption" | "contradiction",
      "requires_research": true | false,
      "suggested_query": "<optional query if research required>"
    }
  ],
  "probabilities": {
    "supported": 0.40,
    "refuted": 0.30,
    "unknown": 0.30
  },
  "epistemic_assessment": "<2-sentence critical evaluation of the debate state>"
}"""


def socratic_critic_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Audits logical validity, generates Socratic probes,
    requests targeted research if critical gaps exist, and outputs a 4th belief vector.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        return {
            "error": "Socratic Critic: Target claim not found in state.",
            "audit_trail": [format_audit_log("Socratic Critic", "Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    evidence_text = evidence_pool.format_pool_for_prompt(claim_id)

    # Compile reasoning transcript so far
    positions = [p for p in state.get("agent_positions", []) if p.claim_id == claim_id]
    debate_summary = "\n".join(
        f"[{p.agent_name.upper()}]: {p.reasoning} (Weaknesses noted: {', '.join(p.acknowledged_weaknesses) or 'None'})"
        for p in positions
    ) or "No debate turns recorded."

    prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Retrieved Evidence:
{evidence_text}

Debate Arguments Under Audit:
{debate_summary}

Perform your Socratic critique."""

    llm = get_llm("critic", temperature=0.1)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        challenges, bv, assessment, research_queries = _parse_critic_response(
            response.content, claim_id, state.get("current_round", 0)
        )
    except Exception as e:
        challenges, bv, assessment, research_queries = _fallback_critic(
            claim_id, state.get("current_round", 0)
        )

    # Record position for Critic in agent_positions
    critic_position = AgentPosition(
        agent_name="critic",
        claim_id=claim_id,
        round_number=state.get("current_round", 0),
        position="SUPPORTED" if bv.supported >= 0.5 else ("REFUTED" if bv.refuted >= 0.5 else "NOT_ENOUGH_INFORMATION"),
        probabilities=bv,
        cited_evidence_ids=[],
        reasoning=assessment,
        acknowledged_weaknesses=["Critic focuses on vulnerabilities rather than compiling new affirmative proof."],
        confidence=max(bv.supported, bv.refuted, bv.unknown),
    )

    req_count = len(research_queries)
    log_msg = format_audit_log(
        "Socratic Critic",
        f"Issued {len(challenges)} probe(s) ({req_count} research request(s)). P(Supp)={bv.supported:.2f}, P(Ref)={bv.refuted:.2f}"
    )

    return {
        "agent_positions": [critic_position],
        "critic_challenges": challenges,
        "critic_research_requests": research_queries,
        "belief_vectors": {"critic": bv},
        "audit_trail": [log_msg],
        "error": None,
    }


def _parse_critic_response(raw: str, claim_id: str, round_num: int):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    raw_challenges = data.get("challenges", [])
    challenges: List[SocraticChallenge] = []
    research_queries: List[str] = []

    for idx, c in enumerate(raw_challenges, 1):
        sc = SocraticChallenge(
            challenge_id=c.get("challenge_id", f"SC{idx}"),
            claim_id=claim_id,
            target_agent=c.get("target_agent", "all"),
            question=c.get("question", "What direct evidence proves this assertion?"),
            challenge_type=c.get("challenge_type", "unsupported_assumption"),
            requires_research=bool(c.get("requires_research", False)),
            suggested_query=c.get("suggested_query"),
        )
        challenges.append(sc)
        if sc.requires_research and sc.suggested_query:
            research_queries.append(sc.suggested_query)

    probs_dict = data.get("probabilities", {"supported": 0.4, "refuted": 0.3, "unknown": 0.3})
    bv = BeliefVector(
        supported=float(probs_dict.get("supported", 0.4)),
        refuted=float(probs_dict.get("refuted", 0.3)),
        unknown=float(probs_dict.get("unknown", 0.3)),
    )
    assessment = data.get("epistemic_assessment", "Socratic audit highlighted core premises and evidentiary dependencies.")
    return challenges, bv, assessment, research_queries


def _fallback_critic(claim_id: str, round_num: int):
    challenges = [
        SocraticChallenge(
            challenge_id="SC1",
            claim_id=claim_id,
            target_agent="all",
            question="Does the primary documentation confirm the exact parameters of the claim without reliance on secondary interpretations?",
            challenge_type="temporal_drift",
            requires_research=False,
            suggested_query=None,
        )
    ]
    bv = BeliefVector(supported=0.45, refuted=0.25, unknown=0.30)
    assessment = "Debate demonstrates plausible evidence; scrutiny required for chronological precision."
    return challenges, bv, assessment, []
