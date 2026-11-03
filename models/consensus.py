from pydantic import BaseModel, Field
from typing import List, Dict
from models.agents import BeliefVector


class DeGrootIteration(BaseModel):
    """
    State of beliefs at iteration t of DeGroot consensus.
    """
    iteration: int
    agent_beliefs: Dict[str, List[float]] = Field(
        ...,
        description="Belief vector [p_supp, p_ref, p_unk] for each agent"
    )
    mean_belief: List[float]
    max_disagreement: float = Field(..., description="Max variation across agent beliefs at this iteration")


class ConsensusResult(BaseModel):
    """
    Result of the DeGroot mathematical consensus process.
    """
    claim_id: str
    round_number: int
    converged: bool
    iterations_run: int
    final_consensus: BeliefVector
    history: List[DeGrootIteration] = Field(default_factory=list)
    trust_matrix: List[List[float]] = Field(default_factory=list)
    agent_names: List[str] = Field(default_factory=list)
