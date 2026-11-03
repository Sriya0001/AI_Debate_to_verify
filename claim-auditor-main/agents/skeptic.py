from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from config import config
from graph.state import AuditState, EvidenceItem, DebateTurn, EvidenceType
from tools.search import search, format_results_for_prompt


# Prompts

SYSTEM_PROMPT = """You are the Skeptic in a fact-checking debate. Your sole 
job is to build the strongest possible case that a given claim is FALSE, 
misleading, or insufficiently supported.

Look for: factual inaccuracies, missing context, outdated information, 
overgeneralizations, unsupported causal claims, or cherry-picked data.

Your response must have exactly two parts, separated by the marker ---ARGUMENT---:

Part 1: A 2-3 sentence argument challenging the claim, citing specific 
evidence from the search results. Be concrete — reference actual sources.
If you find the claim is well-supported despite your search, say so honestly.

Part 2: A single float between 0.0 and 1.0 representing how confident you are 
the claim is FALSE or problematic. 0.0 = claim seems solid, 
1.0 = claim is clearly wrong. Be calibrated.

Format exactly:
<your argument here>
---ARGUMENT---
<confidence float>"""


USER_PROMPT_TEMPLATE = """Claim to challenge:
\"{claim_text}\"

Web search results:
{search_results}

Debate round: {current_round}
{prior_advocate_argument}"""


# Node

def skeptic_node(state: AuditState) -> dict:
    """
    LangGraph node: build a challenging argument against the current claim.

    Reads:  state["pending_claim_ids"], state["claims"],
            state["debate_transcript"], state["current_round"]
    Writes: state["evidence"] (appends), state["debate_transcript"] (appends)
            state["error"] on failure
    """
    current_claim = _get_current_claim(state)
    if current_claim is None:
        return {"error": "Skeptic could not find the current pending claim."}

    # Search for contradicting evidence
    search_query = f"evidence against or counterargument: {current_claim['text']}"
    try:
        results = search(search_query)
    except Exception as e:
        return {"error": f"Skeptic search failed: {e}"}

    formatted_results = format_results_for_prompt(results)

    # Pull prior advocate argument for rebuttal context
    prior_advocate = _get_prior_argument(
        state, claim_id=current_claim["id"], agent="advocate"
    )
    prior_block = (
        f"The advocate previously argued:\n\"{prior_advocate}\"\nChallenge this."
        if prior_advocate else ""
    )

    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT_TEMPLATE.format(
            claim_text=current_claim["text"],
            search_results=formatted_results,
            current_round=state["current_round"],
            prior_advocate_argument=prior_block,
        )),
    ]

    try:
        response = llm.invoke(messages)
        argument, confidence = _parse_response(response.content)
    except Exception as e:
        return {"error": f"Skeptic LLM call failed: {e}"}

    evidence_items: list[EvidenceItem] = []
    for r in results:
        evidence_items.append(EvidenceItem(
            claim_id=current_claim["id"],
            content=r["content"],
            source_url=r.get("url"),
            evidence_type=EvidenceType.WEB_SEARCH,
            stance="contradicting",
            agent="skeptic",
        ))

    debate_turn = DebateTurn(
        claim_id=current_claim["id"],
        round=state["current_round"],
        agent="skeptic",
        argument=argument,
    )

    return {
        "evidence":          evidence_items,
        "debate_transcript": [debate_turn],
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


def _get_prior_argument(
    state: AuditState,
    claim_id: str,
    agent: str,
) -> str | None:
    turns = [
        t for t in state["debate_transcript"]
        if t["claim_id"] == claim_id and t["agent"] == agent
    ]
    return turns[-1]["argument"] if turns else None


def _parse_response(raw: str) -> tuple[str, float]:
    parts = raw.strip().split("---ARGUMENT---")
    if len(parts) != 2:
        raise ValueError(
            f"Skeptic response missing ---ARGUMENT--- separator. Got: {raw[:200]}"
        )

    argument = parts[0].strip()
    conf_str = parts[1].strip()

    try:
        confidence = float(conf_str)
    except ValueError:
        import re
        match = re.search(r"\d+\.\d+|\d+", conf_str)
        confidence = float(match.group()) if match else 0.5

    confidence = max(0.0, min(1.0, confidence))
    return argument, confidence


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=config.skeptic_model,
        api_key=config.anthropic_api_key,
        temperature=0.3,
    )