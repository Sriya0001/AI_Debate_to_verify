from pydantic import BaseModel, Field
from graph.state import Ruling


# ── Request ───────────────────────────────────────────────────────────────────

class AuditRequest(BaseModel):
    text:         str            = Field(..., description="The text to audit")
    source_label: str | None     = Field(None, description="e.g. 'GPT-4o response'")
    max_rounds:   int            = Field(2, ge=1, le=5)


# Response sub-objects

class ClaimVerdictResponse(BaseModel):
    claim_id:      str
    claim_text:    str           # joined from claims list for convenience
    ruling:        Ruling
    confidence:    float
    reasoning:     str
    debate_rounds: int


class AuditResponse(BaseModel):
    # Top-level result
    aggregate_verdict:    Ruling
    aggregate_confidence: float
    audit_summary:        str
    escalate_to_human:    bool

    # Per-claim detail
    claim_verdicts: list[ClaimVerdictResponse]

    # Metadata
    total_claims:   int
    source_label:   str | None


# Error response

class ErrorResponse(BaseModel):
    error: str