from typing import TypedDict, Annotated
from enum import Enum
import operator


# Enums

class Ruling(str, Enum):
    SUPPORTED  = "supported"
    REFUTED    = "refuted"
    UNCERTAIN  = "uncertain"
    ESCALATED  = "escalated"   # human review needed


class EvidenceType(str, Enum):
    WEB_SEARCH    = "web_search"
    GROUNDED_RAG  = "grounded_rag"
    REASONING     = "reasoning"    # agent's own logical inference, no retrieval


# Sub-objects

class Claim(TypedDict):
    id:            str          # e.g. "c1", "c2"
    text:          str          # the atomic claim as extracted
    source_span:   str          # substring of original text this came from


class EvidenceItem(TypedDict):
    claim_id:      str
    content:       str          # the evidence text
    source_url:    str | None   # if retrieved from web
    evidence_type: EvidenceType
    stance:        str          # "supporting" | "contradicting"
    agent:         str          # "advocate" | "skeptic"


class DebateTurn(TypedDict):
    claim_id:      str
    round:         int
    agent:         str          # "advocate" | "skeptic"
    argument:      str


class ClaimVerdict(TypedDict):
    claim_id:      str
    ruling:        Ruling
    confidence:    float        # 0.0 – 1.0
    reasoning:     str          # arbitrator's explanation
    debate_rounds: int          # how many rounds it took


# Top-level graph state

class AuditState(TypedDict):

    # Input
    input_text:        str              # the LLM output being audited
    source_label:      str | None       # e.g. "GPT-4o response", "RAG answer"

    # Decomposition
    claims:            list[Claim]      # populated by decomposer node
    pending_claim_ids: list[str]        # claims not yet fully processed
                                        # shrinks as graph progresses

    # Evidence
    # Annotated with operator.add so multiple agents can append
    # to the same list without overwriting each other's work
    evidence:          Annotated[list[EvidenceItem], operator.add]

    # Debate
    debate_transcript: Annotated[list[DebateTurn], operator.add]
    current_round:     int              # which debate round we're in
    max_rounds:        int              # circuit breaker; default 2

    # Verdicts
    claim_verdicts:       list[ClaimVerdict]
    aggregate_verdict:    Ruling | None    # overall ruling across all claims
    aggregate_confidence: float | None     # mean confidence, weighted by claim length
    audit_summary:        str | None       # narrative summary produced by reporter

    # Control flow
    escalate_to_human: bool             # arbitrator sets True when confidence < threshold
    audit_complete:    bool             # reporter sets True when done
    # Annotated so parallel nodes (advocate + skeptic) can both write errors
    # without triggering INVALID_CONCURRENT_GRAPH_UPDATE
    error: Annotated[str | None, lambda a, b: "; ".join(filter(None, [a, b])) or None]


# Factory

def initial_state(input_text: str, source_label: str | None = None) -> AuditState:
    """
    Build a clean starting state for a new audit run.
    Call this from the API layer before invoking the graph.
    """
    return AuditState(
        input_text=input_text,
        source_label=source_label,
        claims=[],
        pending_claim_ids=[],
        evidence=[],
        debate_transcript=[],
        current_round=0,
        max_rounds=2,
        claim_verdicts=[],
        aggregate_verdict=None,
        aggregate_confidence=None,
        audit_summary=None,
        escalate_to_human=False,
        audit_complete=False,
        error=None,
    )