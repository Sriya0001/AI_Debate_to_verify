from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

from graph.state import VeritasState
from graph.routing import route_after_consensus, route_after_verifier, route_after_pending
from agents.claim_analyzer import claim_analyzer_node
from agents.planner import planner_node
from agents.evidence_retriever import evidence_retriever_node
from agents.source_credibility import source_credibility_node
from agents.advocate import advocate_node
from agents.skeptic import skeptic_node
from agents.neutral import neutral_analyst_node
from agents.critic import socratic_critic_node
from agents.judge import judge_node
from agents.final_verifier import final_verifier_node
from agents.reporter import reporter_node, check_pending_node
from consensus.node import degroot_consensus_node
from utils.audit_trail import format_audit_log


def autonomous_research_node(state: VeritasState) -> Dict[str, Any]:
    """
    Activated when the Autonomous Router identifies insufficient evidence,
    source contradictions, or Socratic research requests. Formulates deeper queries
    and increments the research_rounds counter.
    """
    claim_id = state["current_claim"].claim_id if state.get("current_claim") else "C1"
    r_round = state.get("research_rounds", 0) + 1

    log_msg = format_audit_log(
        "Autonomous Planner",
        f"Initiating research loop {r_round}: deeper query reformulation for claim {claim_id}"
    )

    # Re-run planner to formulate deeper/alternative queries
    return {
        "research_rounds": r_round,
        "critic_research_requests": [],
        "audit_trail": [log_msg],
    }


def debate_turn_increment_node(state: VeritasState) -> Dict[str, Any]:
    """Increments debate round counter when re-entering debate loop."""
    return {"current_round": state.get("current_round", 0) + 1}


def build_veritas_graph() -> StateGraph:
    """
    Builds and compiles the full VERITAS Autonomous Multi-Agent verification graph.
    """
    builder = StateGraph(VeritasState)

    # ── Add All 10 Agent Nodes + Graph Support Nodes ─────────────────────────
    builder.add_node("claim_analyzer", claim_analyzer_node)
    builder.add_node("planner", planner_node)
    builder.add_node("evidence_retriever", evidence_retriever_node)
    builder.add_node("source_credibility", source_credibility_node)
    builder.add_node("advocate", advocate_node)
    builder.add_node("skeptic", skeptic_node)
    builder.add_node("neutral", neutral_analyst_node)
    builder.add_node("critic", socratic_critic_node)
    builder.add_node("judge", judge_node)
    builder.add_node("degroot_consensus", degroot_consensus_node)
    builder.add_node("autonomous_research", autonomous_research_node)
    builder.add_node("final_verifier", final_verifier_node)
    builder.add_node("check_pending", check_pending_node)
    builder.add_node("reporter", reporter_node)

    # ── Define Deterministic Edges ──────────────────────────────────────────
    # 1. Start -> Claim Analyzer -> Planner -> Evidence Retrieval
    builder.add_edge(START, "claim_analyzer")
    builder.add_edge("claim_analyzer", "planner")
    builder.add_edge("planner", "evidence_retriever")
    builder.add_edge("evidence_retriever", "source_credibility")

    # 2. Source Credibility fans out in parallel to Advocate, Skeptic, and Neutral Analyst
    builder.add_edge("source_credibility", "advocate")
    builder.add_edge("source_credibility", "skeptic")
    builder.add_edge("source_credibility", "neutral")

    # 3. Advocate, Skeptic, and Neutral fan-in to Socratic Critic
    builder.add_edge(["advocate", "skeptic", "neutral"], "critic")

    # 4. Critic -> Judge -> DeGroot Consensus
    builder.add_edge("critic", "judge")
    builder.add_edge("judge", "degroot_consensus")

    # ── Conditional Edges ───────────────────────────────────────────────────
    # 5. After DeGroot Consensus: Autonomous Router decides next action
    builder.add_conditional_edges(
        "degroot_consensus",
        route_after_consensus,
        {
            "autonomous_research": "autonomous_research",
            "advocate": "advocate",
            "skeptic": "skeptic",
            "neutral": "neutral",
            "final_verifier": "final_verifier",
        }
    )

    # 6. Autonomous Research loops back to Evidence Retrieval
    builder.add_edge("autonomous_research", "evidence_retriever")

    # 7. After Final Verifier: check if more claims remain
    builder.add_conditional_edges(
        "final_verifier",
        route_after_verifier,
        {
            "check_pending": "check_pending",
            "reporter": "reporter",
        }
    )

    # 8. Check Pending loops back to Planner for next claim or wraps up in Reporter
    builder.add_conditional_edges(
        "check_pending",
        route_after_pending,
        {
            "planner": "planner",
            "reporter": "reporter",
        }
    )

    # 9. Reporter terminates graph
    builder.add_edge("reporter", END)

    return builder.compile()


# Module-level compiled graph
veritas_graph = build_veritas_graph()
