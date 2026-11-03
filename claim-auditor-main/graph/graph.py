from langgraph.graph import StateGraph, START, END

from graph.state import AuditState, Ruling


# Node imports
# Each agent module exposes a single callable: node_fn(state) -> dict
# The dict contains only the keys being updated — LangGraph merges it
# into the existing state rather than replacing it wholesale.

from agents.decomposer  import decomposer_node
from agents.advocate    import advocate_node
from agents.skeptic     import skeptic_node
from agents.arbitrator  import arbitrator_node
from agents.reporter    import reporter_node


# Conditional edge functions
    # These are plain Python functions — not nodes.
    # They read state and return a string key that LangGraph uses to route.

def route_after_arbitrator(state: AuditState) -> str | list[str]:
    """
    After the arbitrator sets a verdict for the current claim, decide:
      - If we haven't hit max_rounds and confidence is low → fan out to
        both advocate AND skeptic for another debate round on the same claim.
      - Otherwise → check if there are more claims to process.

    Must return both advocate and skeptic when looping because the join edge
    add_edge(["advocate", "skeptic"], "arbitrator") requires both to complete
    before arbitrator can run again.
    """
    low_confidence = (
        state["claim_verdicts"]
        and state["claim_verdicts"][-1]["confidence"] < 0.5
    )
    rounds_remaining = state["current_round"] < state["max_rounds"]

    if low_confidence and rounds_remaining:
        return ["advocate", "skeptic"]  # another round — both agents debate
    return "check_pending"              # move on


def route_after_pending_check(state: AuditState) -> str | list[str]:
    """
    After a claim is fully resolved, decide:
      - More claims still in the queue → fan out to both advocate AND skeptic
        for the next claim.
      - Queue empty → send to reporter to wrap up.

    Must return both advocate and skeptic when looping — same join requirement
    as above.
    """
    if state["pending_claim_ids"]:
        return ["advocate", "skeptic"]  # start fresh on next claim — both debate
    return "reporter"


# Graph builder 

def build_graph() -> StateGraph:
    """
    Construct and compile the ClaimAuditor LangGraph.

    Node sequence (happy path, single claim, single round):
        START → decomposer → advocate + skeptic (parallel) →
        arbitrator → [check_pending] → reporter → END

    With loops:
        arbitrator → advocate (low confidence, rounds remain)
        check_pending → advocate (more claims in queue)
    """
    builder = StateGraph(AuditState)

    # Register nodes 
    builder.add_node("decomposer",     decomposer_node)
    builder.add_node("advocate",       advocate_node)
    builder.add_node("skeptic",        skeptic_node)
    builder.add_node("arbitrator",     arbitrator_node)
    builder.add_node("check_pending",  _check_pending_node)
    builder.add_node("reporter",       reporter_node)

    # Fixed edges 
    builder.add_edge(START,          "decomposer")

    # Decomposer fans out to both advocate AND skeptic simultaneously.
    # LangGraph runs them in parallel; operator.add on evidence and
    # debate_transcript merges their outputs automatically.
    builder.add_edge("decomposer",   "advocate")
    builder.add_edge("decomposer",   "skeptic")

    # Both agents must finish before arbitrator runs.
    # add_edge with a list of sources means "wait for ALL of these".
    builder.add_edge(["advocate", "skeptic"], "arbitrator")

    builder.add_edge("reporter",     END)

    # Conditional edges
    builder.add_conditional_edges(
        "arbitrator",
        route_after_arbitrator,
        {
            "advocate":      "advocate",        # loop: another debate round (fan-out)
            "skeptic":       "skeptic",         # loop: another debate round (fan-out)
            "check_pending": "check_pending",   # proceed: claim resolved
        }
    )

    builder.add_conditional_edges(
        "check_pending",
        route_after_pending_check,
        {
            "advocate": "advocate",     # loop: next claim (fan-out)
            "skeptic":  "skeptic",      # loop: next claim (fan-out)
            "reporter": "reporter",     # done: all claims resolved
        }
    )

    return builder.compile()


# Internal helper node

def _check_pending_node(state: AuditState) -> dict:
    """
    Lightweight node that sits between arbitrator and the pending check edge.
    Pops the just-processed claim from pending_claim_ids and resets
    current_round to 0 so the next claim gets a fresh debate.

    Returns only the keys it changes — LangGraph merges the rest.
    """
    remaining = list(state["pending_claim_ids"])

    # Pop the first pending claim (the one we just finished)
    if remaining:
        remaining.pop(0)

    return {
        "pending_claim_ids": remaining,
        "current_round":     0,          # reset for next claim's debate
    }


# Module-level compiled graph
# Import this singleton from the API layer and UI.
# Building once at import time means the graph structure is validated
# immediately on startup rather than at first request.

graph = build_graph()