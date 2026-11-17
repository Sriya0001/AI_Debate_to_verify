from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

from graph.state import VeritasState
from graph.routing import (
    route_after_consensus,
    route_after_verifier,
    route_after_pending,
    route_pipeline_mode,
)
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
from consensus.fcmad_node import (
    fcmad_debaters_round0_node,
    fcmad_debaters_revise_node,
    fcmad_degroot_node,
    fcmad_pool_node,
    route_fcmad_stop,
)
from agents.fcmad_judge import fcmad_judge_node
from agents.fcmad_summarizer import fcmad_summarizer_node
from agents.fcmad_output import fcmad_output_node
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
    """
    Increments debate round counter when re-entering debate loop.
    Note: In the standard VERITAS graph, current_round is incremented inside
    degroot_consensus_node (consensus/node.py) at the completion of each round.
    This helper is preserved for custom subgraphs or backward compatibility.
    """
    return {"current_round": state.get("current_round", 0) + 1}


def build_veritas_graph() -> StateGraph:
    """
    Builds and compiles the unified FC-MAD Autonomous Multi-Agent verification graph
    (Mathematical DeGroot Consensus with Total Variation Distance).
    """
    builder = StateGraph(VeritasState)

    # ── Core Pipeline Nodes ───────────────────────────────────────────────────
    builder.add_node("claim_analyzer", claim_analyzer_node)
    builder.add_node("planner", planner_node)
    builder.add_node("evidence_retriever", evidence_retriever_node)
    builder.add_node("source_credibility", source_credibility_node)
    builder.add_node("check_pending", check_pending_node)
    builder.add_node("reporter", reporter_node)

    # ── FC-MAD Multi-Agent Deliberation & DeGroot TV-Consensus Nodes ─────────
    builder.add_node("fcmad_debaters_round0", fcmad_debaters_round0_node)
    builder.add_node("fcmad_judge", fcmad_judge_node)
    builder.add_node("fcmad_summarizer", fcmad_summarizer_node)
    builder.add_node("fcmad_debaters_revise", fcmad_debaters_revise_node)
    builder.add_node("fcmad_degroot", fcmad_degroot_node)
    builder.add_node("fcmad_pool", fcmad_pool_node)
    builder.add_node("fcmad_output", fcmad_output_node)

    # ── Edge Wiring ─────────────────────────────────────────────────────────
    # 1. Start -> Claim Analyzer -> Planner -> Evidence Retrieval -> Source Credibility
    builder.add_edge(START, "claim_analyzer")
    builder.add_edge("claim_analyzer", "planner")
    builder.add_edge("planner", "evidence_retriever")
    builder.add_edge("evidence_retriever", "source_credibility")

    # 2. Source Credibility flows directly into FC-MAD Parallel Debaters Round 0
    builder.add_edge("source_credibility", "fcmad_debaters_round0")

    # 3. Round 0 Debaters -> FC-MAD Judge (argument quality q_i & peer credibility c_i)
    builder.add_edge("fcmad_debaters_round0", "fcmad_judge")

    # 4. FC-MAD Stop Router: stops on epsilon consensus (E <= epsilon) or max rounds (t = M)
    builder.add_conditional_edges(
        "fcmad_judge",
        route_fcmad_stop,
        {
            "fcmad_pool": "fcmad_pool",
            "fcmad_summarizer": "fcmad_summarizer",
        }
    )

    # 5. Iteration loop: Summarizer -> Debaters Revise -> DeGroot TV Update -> Judge
    builder.add_edge("fcmad_summarizer", "fcmad_debaters_revise")
    builder.add_edge("fcmad_debaters_revise", "fcmad_degroot")
    builder.add_edge("fcmad_degroot", "fcmad_judge")

    # 6. Final Pooling (p* = sum c_i * p_i), calibration, and abstention check -> Output Agent
    builder.add_edge("fcmad_pool", "fcmad_output")

    # 7. Output Agent -> Check Pending claims or Reporter
    builder.add_conditional_edges(
        "fcmad_output",
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


# Module-level compiled graph (Unified FC-MAD DeGroot consensus architecture)
veritas_graph = build_veritas_graph()
fcmad_graph = veritas_graph
