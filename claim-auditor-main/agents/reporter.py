from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from config import config
from graph.state import AuditState, Ruling


# Prompts

SYSTEM_PROMPT = """You are the Reporter in a fact-checking system. You have 
received the results of a structured multi-agent debate that audited a passage 
of text claim by claim.

Your job is to write a concise, plain-English audit summary that a non-expert 
could read and understand. Do not use jargon.

Structure your response as follows:

OVERALL: One sentence verdict on the passage as a whole.

BREAKDOWN: One bullet per claim in this format:
  • [SUPPORTED/REFUTED/UNCERTAIN] "<claim text>" — <one sentence explanation>

CAVEATS: One or two sentences noting any limitations, contested claims, 
or areas where a human should follow up. If nothing warrants a caveat, 
write "None."

Write clearly and directly. Do not hedge everything — if the evidence is 
strong, say so."""


USER_PROMPT_TEMPLATE = """Original text audited:
\"{input_text}\"

Claim verdicts:
{verdict_details}

Escalation flag: {escalation_flag}"""


# Node

def reporter_node(state: AuditState) -> dict:
    """
    LangGraph node: synthesise all claim verdicts into a final audit report.

    Reads:  state["claim_verdicts"], state["claims"], state["input_text"],
            state["escalate_to_human"]
    Writes: state["aggregate_verdict"], state["aggregate_confidence"],
            state["audit_complete"]
    
    Note: the narrative summary is stored in the final claim_verdicts reasoning
    fields — the reporter's LLM output goes into aggregate fields and a new
    "summary" key we add to state here.
    """
    if not state["claim_verdicts"]:
        return {
            "error":             "Reporter received no claim verdicts to summarise.",
            "audit_complete":    False,
        }

    # Compute aggregate verdict and confidence
    aggregate_verdict    = _compute_aggregate_verdict(state)
    aggregate_confidence = _compute_aggregate_confidence(state)

    # Build verdict details for the prompt
    verdict_details = _format_verdict_details(state)
    escalation_flag = (
        "One or more claims require human review due to low confidence "
        "or genuine contestation."
        if state["escalate_to_human"]
        else "No escalation required."
    )

    # Call LLM for narrative summary
    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT_TEMPLATE.format(
            input_text=state["input_text"],
            verdict_details=verdict_details,
            escalation_flag=escalation_flag,
        )),
    ]

    try:
        response = llm.invoke(messages)
        narrative = response.content.strip()
    except Exception as e:
        # Reporter failure is non-fatal — we still have structured verdicts.
        # Return what we have and flag the error in the narrative.
        narrative = f"[Reporter narrative generation failed: {e}]"

    return {
        "aggregate_verdict":    aggregate_verdict,
        "aggregate_confidence": aggregate_confidence,
        "audit_summary":        narrative,
        "audit_complete":       True,
    }


# Helpers

def _compute_aggregate_verdict(state: AuditState) -> Ruling:
    """
    Roll up individual claim rulings into one overall verdict.

    Logic:
      - Any REFUTED claim → aggregate is REFUTED
        (one clear falsehood poisons the passage)
      - Any ESCALATED claim → aggregate is ESCALATED
      - All SUPPORTED → SUPPORTED
      - Otherwise → UNCERTAIN
    """
    rulings = [v["ruling"] for v in state["claim_verdicts"]]

    if Ruling.REFUTED in rulings:
        return Ruling.REFUTED
    if Ruling.ESCALATED in rulings:
        return Ruling.ESCALATED
    if all(r == Ruling.SUPPORTED for r in rulings):
        return Ruling.SUPPORTED
    return Ruling.UNCERTAIN


def _compute_aggregate_confidence(state: AuditState) -> float:
    """
    Weighted mean confidence across all claims.
    Weight by claim text length as a rough proxy for claim complexity —
    a longer claim that was well-supported should count for more than
    a trivial short claim.
    """
    if not state["claim_verdicts"]:
        return 0.0

    # Build a lookup from claim_id → claim text length
    claim_lengths = {c["id"]: len(c["text"]) for c in state["claims"]}
    total_length  = sum(claim_lengths.values()) or 1   # avoid div-by-zero

    weighted_sum = 0.0
    for v in state["claim_verdicts"]:
        weight        = claim_lengths.get(v["claim_id"], 1) / total_length
        weighted_sum += v["confidence"] * weight

    return round(weighted_sum, 3)


def _format_verdict_details(state: AuditState) -> str:
    """
    Build a readable verdict block for injection into the reporter prompt.
    Joins each claim's text with its ruling, confidence, and reasoning.
    """
    # Build claim text lookup
    claim_text = {c["id"]: c["text"] for c in state["claims"]}

    lines = []
    for v in state["claim_verdicts"]:
        text      = claim_text.get(v["claim_id"], "[unknown claim]")
        ruling    = v["ruling"].upper()
        conf_pct  = int(v["confidence"] * 100)
        reasoning = v["reasoning"]

        lines.append(
            f"Claim: \"{text}\"\n"
            f"Ruling: {ruling} (confidence: {conf_pct}%)\n"
            f"Reasoning: {reasoning}"
        )

    return "\n\n".join(lines)


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=config.reporter_model,
        api_key=config.anthropic_api_key,
        temperature=0.2,
    )