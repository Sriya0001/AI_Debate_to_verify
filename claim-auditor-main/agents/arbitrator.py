import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from config import config
from graph.state import (
    AuditState, ClaimVerdict, Ruling,
    EvidenceItem, DebateTurn,
)


# Config

CONFIDENCE_ESCALATION_THRESHOLD = 0.4   # below this → escalate to human
CONFLICT_THRESHOLD              = 0.7   # both sides above this → contested


# Prompts

SYSTEM_PROMPT = """You are the Arbitrator in a structured fact-checking debate.
You have observed an Advocate and a Skeptic argue about a factual claim.
Your job is to render a fair, calibrated verdict.

Evaluate the QUALITY of arguments and evidence, not just who argued more 
confidently. A single primary source directly addressing the claim is worth 
more than three tangentially related sources. Look for:
  - Whether sources directly address the claim or are obliquely related
  - Whether the advocate or skeptic found stronger, more specific evidence
  - Whether the claim is too vague or context-dependent to verify cleanly
  - Whether both sides found strong but contradictory evidence (genuinely contested)

Your response must be valid JSON only. No preamble, no markdown fences.

Schema:
{
  "ruling":      "supported" | "refuted" | "uncertain",
  "confidence":  <float 0.0-1.0, your confidence in the ruling>,
  "reasoning":   "<2-3 sentences explaining your verdict, citing specific evidence>"
}

Confidence guide:
  0.9-1.0  Strong, direct evidence clearly settles the claim
  0.7-0.9  Good evidence, minor gaps or ambiguity
  0.5-0.7  Moderate evidence, meaningful uncertainty remains
  0.3-0.5  Weak or conflicting evidence, hard to call
  0.0-0.3  Genuinely contested or unverifiable — consider escalating"""


USER_PROMPT_TEMPLATE = """Claim under review:
\"{claim_text}\"

--- ADVOCATE'S ARGUMENTS ---
{advocate_turns}

--- SKEPTIC'S ARGUMENTS ---
{skeptic_turns}

--- EVIDENCE SOURCES FOUND ---
{evidence_summary}

Render your verdict."""


# Node

def arbitrator_node(state: AuditState) -> dict:
    """
    LangGraph node: weigh the debate and produce a ClaimVerdict.

    Reads:  state["pending_claim_ids"], state["claims"],
            state["debate_transcript"], state["evidence"],
            state["current_round"], state["claim_verdicts"]
    Writes: state["claim_verdicts"] (appends),
            state["current_round"] (increments),
            state["escalate_to_human"] (may set True)
            state["error"] on failure
    """
    current_claim = _get_current_claim(state)
    if current_claim is None:
        return {"error": "Arbitrator could not find the current pending claim."}

    claim_id = current_claim["id"]

    # Filter debate and evidence to current claim only
    advocate_turns = [
        t for t in state["debate_transcript"]
        if t["claim_id"] == claim_id and t["agent"] == "advocate"
    ]
    skeptic_turns = [
        t for t in state["debate_transcript"]
        if t["claim_id"] == claim_id and t["agent"] == "skeptic"
    ]
    claim_evidence = [
        e for e in state["evidence"]
        if e["claim_id"] == claim_id
    ]

    # Build prompt context
    advocate_text  = _format_turns(advocate_turns)
    skeptic_text   = _format_turns(skeptic_turns)
    evidence_text  = _format_evidence(claim_evidence)

    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT_TEMPLATE.format(
            claim_text=current_claim["text"],
            advocate_turns=advocate_text,
            skeptic_turns=skeptic_text,
            evidence_summary=evidence_text,
        )),
    ]

    try:
        response = llm.invoke(messages)
        ruling, confidence, reasoning = _parse_response(response.content)
    except Exception as e:
        return {"error": f"Arbitrator LLM call failed: {e}"}

    # Build verdict
    verdict = ClaimVerdict(
        claim_id=claim_id,
        ruling=ruling,
        confidence=confidence,
        reasoning=reasoning,
        debate_rounds=state["current_round"] + 1,
    )

    # Escalation logic
    # Trigger 1: arbitrator itself is not confident
    should_escalate = confidence < CONFIDENCE_ESCALATION_THRESHOLD

    # Trigger 2: both sides argued strongly in opposite directions
    # (genuinely contested — not just weak evidence on both sides)
    if not should_escalate:
        should_escalate = _is_contested(advocate_turns, skeptic_turns)

    # Existing verdicts + new one
    # claim_verdicts does NOT use operator.add, so we manage the full list.
    # Replace any prior verdict for this claim (from earlier rounds) with
    # the latest one — we only want the final verdict per claim.
    existing = [v for v in state["claim_verdicts"] if v["claim_id"] != claim_id]
    updated_verdicts = existing + [verdict]

    return {
        "claim_verdicts":    updated_verdicts,
        "current_round":     state["current_round"] + 1,
        "escalate_to_human": should_escalate,
    }


# Helpers

def _get_current_claim(state: AuditState) -> dict | None:
    if not state["pending_claim_ids"]:
        return None
    current_id = state["pending_claim_ids"][0]
    return next(
        (c for c in state["claims"] if c["id"] == current_id),
        None
    )


def _format_turns(turns: list[DebateTurn]) -> str:
    if not turns:
        return "No arguments recorded."
    lines = []
    for t in turns:
        lines.append(f"[Round {t['round']}] {t['argument']}")
    return "\n\n".join(lines)


def _format_evidence(evidence: list[EvidenceItem]) -> str:
    """
    Summarise evidence sources for the prompt.
    We show URLs and stance so the arbitrator can weight source quality.
    We don't dump full content — that would blow the context window fast.
    """
    if not evidence:
        return "No evidence retrieved."

    supporting    = [e for e in evidence if e["stance"] == "supporting"]
    contradicting = [e for e in evidence if e["stance"] == "contradicting"]

    lines = []

    if supporting:
        lines.append("Supporting sources:")
        for e in supporting:
            url = e["source_url"] or "no url"
            lines.append(f"  + {url}")

    if contradicting:
        lines.append("Contradicting sources:")
        for e in contradicting:
            url = e["source_url"] or "no url"
            lines.append(f"  - {url}")

    return "\n".join(lines)


def _is_contested(
    advocate_turns: list[DebateTurn],
    skeptic_turns:  list[DebateTurn],
) -> bool:
    """
    Heuristic: if both sides have argued multiple rounds without conceding,
    the claim is genuinely contested and warrants human review.
    Not perfect — a future version could parse confidence from each turn.
    """
    return len(advocate_turns) >= 2 and len(skeptic_turns) >= 2


def _parse_response(raw: str) -> tuple[Ruling, float, str]:
    """
    Parse the arbitrator's JSON response into (ruling, confidence, reasoning).
    """
    # Strip accidental markdown fences
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
        clean = clean.rsplit("```", 1)[0]

    data = json.loads(clean)

    ruling_str = data["ruling"].lower().strip()
    try:
        ruling = Ruling(ruling_str)
    except ValueError:
        # Fallback if LLM returns something slightly off
        ruling = Ruling.UNCERTAIN

    confidence = float(data["confidence"])
    confidence = max(0.0, min(1.0, confidence))

    reasoning = data["reasoning"].strip()

    return ruling, confidence, reasoning


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=config.arbitrator_model,
        api_key=config.anthropic_api_key,
        temperature=0,
    )