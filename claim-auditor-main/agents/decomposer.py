import json
import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from config import config
from graph.state import AuditState, Claim


# Prompt

SYSTEM_PROMPT = """You are a claim decomposition expert. Your job is to break 
a passage of text into atomic, independently verifiable factual claims.

Rules:
- Each claim must assert exactly ONE fact. Split compound statements.
- Each claim must be a complete sentence that stands alone without context.
- Preserve the original meaning — do not paraphrase so heavily that the 
  meaning shifts.
- Omit opinions, value judgments, and purely subjective statements. 
  Only include claims that could in principle be verified as true or false.
- For each claim, copy the shortest substring from the original text that 
  the claim is derived from into source_span.

Respond with a JSON array only. No preamble, no explanation, no markdown fences.

Schema:
[
  {
    "text": "<the atomic claim as a complete sentence>",
    "source_span": "<verbatim substring from the original text>"
  },
  ...
]"""


USER_PROMPT_TEMPLATE = """Decompose the following text into atomic claims:

\"\"\"
{input_text}
\"\"\""""


# Node

def decomposer_node(state: AuditState) -> dict:
    """
    LangGraph node: decompose input_text into atomic Claim objects.

    Reads:  state["input_text"]
    Writes: state["claims"], state["pending_claim_ids"]
            state["error"] on failure
    """
    llm = _get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT_TEMPLATE.format(
            input_text=state["input_text"]
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        claims = _parse_claims(raw)
    except json.JSONDecodeError as e:
        return {"error": f"Decomposer failed to parse LLM output as JSON: {e}"}
    except Exception as e:
        return {"error": f"Decomposer encountered an unexpected error: {e}"}

    if not claims:
        return {"error": "Decomposer found no verifiable claims in the input."}

    claim_ids = [c["id"] for c in claims]

    return {
        "claims":            claims,
        "pending_claim_ids": claim_ids,
        "error":             None,
    }


# Helpers

def _parse_claims(raw: str) -> list[Claim]:
    """
    Parse the LLM's JSON response into a list of Claim TypedDicts.
    Assigns a unique ID to each claim here rather than asking the LLM
    to generate IDs — LLMs occasionally produce duplicate or malformed IDs.
    """
    # Strip accidental markdown fences if the LLM adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]         # drop opening fence line
        raw = raw.rsplit("```", 1)[0]       # drop closing fence

    parsed = json.loads(raw)

    claims: list[Claim] = []
    for i, item in enumerate(parsed):
        claim: Claim = {
            "id":          f"c{i+1}",        # c1, c2, c3 ... simple and readable
            "text":        item["text"].strip(),
            "source_span": item["source_span"].strip(),
        }
        claims.append(claim)

    return claims


def _get_llm() -> ChatAnthropic:
    """
    Instantiate the LLM. Centralised here so swapping models
    only requires changing one place per agent.
    Decomposition doesn't need heavy reasoning — gpt-4o-mini is fast and cheap.
    """
    return ChatAnthropic(
        model=config.decomposer_model,
        api_key=config.anthropic_api_key,
        temperature=0,
    )