from langgraph.graph import StateGraph, START, END
from baseline.state import BaselineState, initial_baseline_state
from baseline.agents import (
    baseline_decomposer_node, baseline_advocate_node, baseline_skeptic_node,
    baseline_arbitrator_node, baseline_check_pending_node, baseline_reporter_node
)


def _route_after_arbitrator(state: BaselineState):
    if state["claim_verdicts"] and state["claim_verdicts"][-1]["confidence"] < 0.4 and state["current_round"] < state["max_rounds"]:
        return ["advocate", "skeptic"]
    return "check_pending"


def _route_after_pending(state: BaselineState):
    if state["pending_claim_ids"]:
        return ["advocate", "skeptic"]
    return "reporter"


def build_baseline_graph() -> StateGraph:
    builder = StateGraph(BaselineState)
    builder.add_node("decomposer", baseline_decomposer_node)
    builder.add_node("advocate", baseline_advocate_node)
    builder.add_node("skeptic", baseline_skeptic_node)
    builder.add_node("arbitrator", baseline_arbitrator_node)
    builder.add_node("check_pending", baseline_check_pending_node)
    builder.add_node("reporter", baseline_reporter_node)

    builder.add_edge(START, "decomposer")
    builder.add_edge("decomposer", "advocate")
    builder.add_edge("decomposer", "skeptic")
    builder.add_edge(["advocate", "skeptic"], "arbitrator")

    builder.add_conditional_edges(
        "arbitrator",
        _route_after_arbitrator,
        {
            "advocate": "advocate",
            "skeptic": "skeptic",
            "check_pending": "check_pending",
        }
    )

    builder.add_conditional_edges(
        "check_pending",
        _route_after_pending,
        {
            "advocate": "advocate",
            "skeptic": "skeptic",
            "reporter": "reporter",
        }
    )

    builder.add_edge("reporter", END)
    return builder.compile()


baseline_graph = build_baseline_graph()
