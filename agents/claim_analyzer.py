import json
import re
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.claims import AtomicClaim
from utils.llm import get_llm
from utils.audit_trail import format_audit_log


SYSTEM_PROMPT = """You are the Claim Analysis Agent in VERITAS AI.
Your job is to convert the user's input text into atomic, independently verifiable factual claims.

Rules:
1. Break compound statements into single, independently testable claims.
2. For each claim, identify:
   - claim_id: "C1", "C2", etc.
   - claim_text: Complete, standalone assertion.
   - claim_type: "factual", "numerical", "temporal", or "causal".
   - entities: List of key entities, persons, organizations, or places.
   - temporal_context: Specific timeframe, year, or period mentioned.
   - verification_questions: 1-3 targeted factual questions needed to verify the claim.
   - source_span: The exact substring in the input text from which this claim was extracted.
3. Ignore subjective opinions, greetings, or rhetorical remarks. Focus only on verifiable assertions.

Respond with a JSON array ONLY. No markdown fences, no conversational preamble.

Schema:
[
  {
    "claim_id": "C1",
    "claim_text": "...",
    "claim_type": "factual",
    "entities": ["..."],
    "temporal_context": "...",
    "verification_questions": ["..."],
    "source_span": "..."
  }
]"""


def claim_analyzer_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Decomposes input text into atomic claims.
    """
    input_text = state["input_text"]
    if not input_text or not input_text.strip():
        return {
            "error": "Claim Analyzer received empty input text.",
            "audit_trail": [format_audit_log("Claim Analyzer", "Failed: Empty input text received")],
        }

    llm = get_llm("claim_analyzer", temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Decompose the following text:\n\n\"\"\"\n{input_text}\n\"\"\""),
    ]

    try:
        response = llm.invoke(messages)
        raw_content = response.content.strip()
        claims = _parse_claims_response(raw_content, input_text)
    except Exception as e:
        # Fallback heuristic splitting if LLM or JSON fails
        claims = _fallback_decompose(input_text)

    if not claims:
        claims = [
            AtomicClaim(
                claim_id="C1",
                claim_text=input_text.strip(),
                claim_type="factual",
                entities=[],
                temporal_context=None,
                verification_questions=[f"Is '{input_text.strip()}' supported by reliable evidence?"],
                source_span=input_text.strip(),
            )
        ]

    pending_ids = [c.claim_id for c in claims]
    current_claim = claims[0] if claims else None

    claim_ids_str = ", ".join(pending_ids)
    log_msg = format_audit_log(
        "Claim Analyzer",
        f"Extracted {len(claims)} atomic claim(s): [{claim_ids_str}]"
    )

    return {
        "atomic_claims": claims,
        "pending_claim_ids": pending_ids,
        "current_claim": current_claim,
        "audit_trail": [log_msg],
        "error": None,
    }


def _parse_claims_response(raw: str, original_text: str) -> List[AtomicClaim]:
    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    if not isinstance(data, list):
        data = [data]

    results: List[AtomicClaim] = []
    for idx, item in enumerate(data, 1):
        cid = item.get("claim_id") or f"C{idx}"
        ctext = item.get("claim_text") or item.get("text", "")
        if not ctext.strip():
            continue
        
        results.append(
            AtomicClaim(
                claim_id=cid,
                claim_text=ctext.strip(),
                claim_type=item.get("claim_type", "factual"),
                entities=item.get("entities", []),
                temporal_context=item.get("temporal_context"),
                verification_questions=item.get("verification_questions", []),
                source_span=item.get("source_span", ctext.strip()),
            )
        )
    return results


def _fallback_decompose(text: str) -> List[AtomicClaim]:
    """Heuristic sentence splitter fallback."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
    if not sentences:
        sentences = [text.strip()]

    claims = []
    for idx, s in enumerate(sentences, 1):
        claims.append(
            AtomicClaim(
                claim_id=f"C{idx}",
                claim_text=s,
                claim_type="factual",
                entities=[],
                temporal_context=None,
                verification_questions=[f"Is '{s}' factual?"],
                source_span=s,
            )
        )
    return claims
