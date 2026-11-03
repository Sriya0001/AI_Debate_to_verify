from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ClaimType(str, Enum):
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    GENERAL = "general"


class AtomicClaim(BaseModel):
    """
    Represents an atomic, independently verifiable claim decomposed
    from user input text by the Claim Analysis Agent.
    """
    claim_id: str = Field(..., description="Unique claim identifier, e.g., 'C1', 'C2'")
    claim_text: str = Field(..., description="Self-contained, atomic factual statement")
    claim_type: str = Field("factual", description="Type of claim (factual, numerical, temporal, causal)")
    entities: List[str] = Field(default_factory=list, description="Key named entities, persons, places, organizations")
    temporal_context: Optional[str] = Field(None, description="Temporal anchor or timeframe (e.g. '2024', '1969')")
    verification_questions: List[str] = Field(default_factory=list, description="Key questions required to verify this claim")
    source_span: str = Field("", description="Exact substring from user input corresponding to this claim")


class ClaimAnalysisResult(BaseModel):
    """
    Aggregated result from the Claim Analysis Agent.
    """
    original_text: str
    claims: List[AtomicClaim] = Field(default_factory=list)
