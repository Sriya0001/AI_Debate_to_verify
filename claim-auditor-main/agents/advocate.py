from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from config import config
from graph.state import AuditState, EvidenceItem, DebateTurn, EvidenceType
from tools.search import search, format_results_for_prompt


# Prompts

SYSTEM_PROMPT = """You are the Advocate in a fact-checking debate. Your sole 
job is to build the strongest possible case that a given claim is TRUE.

You have been provided with web search results as your evidence base. Use them.

Your response must have exactly two parts, separated by the marker ---ARGUMENT---:

Part 1: A 2-3 sentence argument defending the claim as true, citing specific 
evidence from the search results. Be concrete — reference actual sources.

Part 2: A single float between 0.0 and 1.0 representing how confident you are 
the claim is true given the evidence you found. 0.0 = certainly false, 
1.0 = certainly true. Be honest — if the evidence is weak, say so with a 
low score even though your job is to advocate.

Format exactly:
<your argument here>
---ARGUMENT---
<confidence float>"""


USER_PROMPT_TEMPLATE = """Claim to defend:
\"{claim_text}\"

Web search results:
{search_results}

Debate round: {current_round}
{prior_skeptic_argument}"""


# Node

def advocate_node(state: AuditState) -> dict:
    """
    LangGraph node: build a supporting argument for the current pending claim.

    Reads:  state["pending_claim_ids"], state["claims"],
            state["debate_transcript"], state["current_round"]
    Writes: state["evidence"] (appends), state["debate_transcript"] (appends)
            state["error"] on failure
    """
    # Identify the current claim
    current_claim = _get_current_claim(state)
    if current_claim is None:
        return {"error": "Advocate could not find the current pending claim."}

    # Search for supporting evidence
    search_query = f"evidence supporting: {current_claim['text']}"
    try:
        results = search(search_query)
    except Exception as e:
        return {"error": f"Advocate search failed: {e}"}

    formatted_results = format_results_for_prompt(results)

    # Build prior skeptic argument for context (round > 0 only)
    prior_skeptic = _get_prior_argument(
        state, claim_id=current_claim["id"], agent="skeptic"
    )
    prior_block = (
        f"The skeptic previously argued:\n\"{prior_skeptic}\"\nRebut this."
        if prior_skeptic else ""
    )

    # Call LLM
    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT_TEMPLATE.format(
            claim_text=current_claim["text"],
            search_results=formatted_results,
            current_round=state["current_round"],
            prior_skeptic_argument=prior_block,
        )),
    ]

    try:
        response = llm.invoke(messages)
        argument, confidence = _parse_response(response.content)
    except Exception as e:
        return {"error": f"Advocate LLM call failed: {e}"}

    # Build output objects
    evidence_items: list[EvidenceItem] = []
    for r in results:
        evidence_items.append(EvidenceItem(
            claim_id=current_claim["id"],
            content=r["content"],
            source_url=r.get("url"),
            evidence_type=EvidenceType.WEB_SEARCH,
            stance="supporting",
            agent="advocate",
        ))

    debate_turn = DebateTurn(
        claim_id=current_claim["id"],
        round=state["current_round"],
        agent="advocate",
        argument=argument,
    )

    return {
        "evidence":          evidence_items,    # operator.add appends
        "debate_transcript": [debate_turn],     # operator.add appends
    }


# Helpers

def _get_current_claim(state: AuditState) -> dict | None:
    """Return the Claim object for the first pending claim ID."""
    if not state["pending_claim_ids"]:
        return None
    current_id = state["pending_claim_ids"][0]
    return next(
        (c for c in state["claims"] if c["id"] == current_id),
        None
    )


def _get_prior_argument(
    state: AuditState,
    claim_id: str,
    agent: str,
) -> str | None:
    """
    Find the most recent debate turn for this claim from the given agent.
    Used so advocate can directly rebut the skeptic's last argument.
    """
    turns = [
        t for t in state["debate_transcript"]
        if t["claim_id"] == claim_id and t["agent"] == agent
    ]
    return turns[-1]["argument"] if turns else None


def _parse_response(raw: str) -> tuple[str, float]:
    """
    Split the LLM response on ---ARGUMENT--- and extract argument + confidence.
    Clamps confidence to [0.0, 1.0] in case the LLM goes out of range.
    """
    parts = raw.strip().split("---ARGUMENT---")
    if len(parts) != 2:
        raise ValueError(
            f"Advocate response missing ---ARGUMENT--- separator. Got: {raw[:200]}"
        )

    argument   = parts[0].strip()
    conf_str   = parts[1].strip()

    try:
        confidence = float(conf_str)
    except ValueError:
        # LLM occasionally adds text after the float — try extracting just digits
        import re
        match = re.search(r"\d+\.\d+|\d+", conf_str)
        if match:
            confidence = float(match.group())
        else:
            confidence = 0.5    # fallback: uncertain

    confidence = max(0.0, min(1.0, confidence))     # clamp
    return argument, confidence


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=config.advocate_model,
        api_key=config.anthropic_api_key,
        temperature=0.3,
    )