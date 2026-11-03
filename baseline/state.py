from typing import TypedDict, Annotated, List, Optional
from enum import Enum
import operator


class BaselineRuling(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"
    ESCALATED = "escalated"


class BaselineClaim(TypedDict):
    id: str
    text: str
    source_span: str


class BaselineEvidenceItem(TypedDict):
    claim_id: str
    content: str
    source_url: Optional[str]
    stance: str
    agent: str


class BaselineDebateTurn(TypedDict):
    claim_id: str
    round: int
    agent: str
    argument: str


class BaselineClaimVerdict(TypedDict):
    claim_id: str
    ruling: BaselineRuling
    confidence: float
    reasoning: str
    debate_rounds: int


class BaselineState(TypedDict):
    input_text: str
    source_label: Optional[str]
    claims: List[BaselineClaim]
    pending_claim_ids: List[str]
    evidence: Annotated[List[BaselineEvidenceItem], operator.add]
    debate_transcript: Annotated[List[BaselineDebateTurn], operator.add]
    current_round: int
    max_rounds: int
    claim_verdicts: List[BaselineClaimVerdict]
    aggregate_verdict: Optional[BaselineRuling]
    aggregate_confidence: Optional[float]
    audit_summary: Optional[str]
    escalate_to_human: bool
    audit_complete: bool
    error: Annotated[Optional[str], lambda a, b: "; ".join(filter(None, [a, b])) or None]


def initial_baseline_state(input_text: str, source_label: Optional[str] = None) -> BaselineState:
    return BaselineState(
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
