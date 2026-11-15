import logging
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.agents import AgentPosition
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log
from config import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Summarizer Agent in the FC-MAD verification system.
Your task is to synthesize the current debate arguments into a concise, objective summary:
1. Key evidence cited by the debaters.
2. Points of agreement between the debaters.
3. Open disputes, doubts, or conflicting interpretations of evidence.

IMPORTANT RULES:
- Do NOT include any numeric probabilities, credibilities, or weights.
- Summarize only the substantive evidentiary arguments.
- Strictly adhere to the brevity limit: Keep the summary under {token_cap} words/tokens."""


def fcmad_summarizer_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Summarizes key evidence, agreements, and disputes across debaters.
    Provides this summary to debaters for subsequent revision rounds without leaking numeric beliefs.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        err_msg = "Summarizer: Target claim not found in state."
        return {
            "error": err_msg,
            "recorded_failures": [{"agent": "summarizer", "error": err_msg}],
            "audit_trail": [format_audit_log("Summarizer", f"Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    current_round = state.get("current_round", 0)

    # Ablation: FC-MAD without Summarizer
    if state.get("fcmad_no_summarizer") or getattr(config, "fcmad_no_summarizer", False):
        log_msg = format_audit_log(
            "Summarizer (FC-MAD Ablation: No Summarizer)",
            "Skipped debate synthesis; revision round debaters reason directly from evidence."
        )
        return {
            "fcmad_debate_summary": "",
            "audit_trail": [log_msg],
            "error": None,
        }

    evidence_text = evidence_pool.format_pool_for_prompt(claim_id)

    # Gather debater positions from the latest round
    positions: List[AgentPosition] = state.get("agent_positions", [])
    claim_positions = [
        p for p in positions
        if p.claim_id == claim_id and p.agent_name.startswith("debater_")
    ]

    # Use the most recent position per debater
    latest_by_debater: Dict[str, AgentPosition] = {}
    for p in claim_positions:
        latest_by_debater[p.agent_name] = p

    debater_arguments = []
    for name, p in sorted(latest_by_debater.items()):
        debater_arguments.append(
            f"- {name.upper()}: Stance={p.position} | Cited={p.cited_evidence_ids}\n  Argument: {p.reasoning}"
        )
    arguments_text = "\n".join(debater_arguments) or "No debater arguments recorded."

    token_cap = getattr(config, "summarizer_token_cap", 300)
    sys_prompt = SYSTEM_PROMPT.format(token_cap=token_cap)

    user_prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Retrieved Evidence Pool:
{evidence_text}

Debater Arguments (Round {current_round}):
{arguments_text}

Produce a concise debate summary highlighting key evidence, agreements, and open disputes (under {token_cap} tokens)."""

    llm = get_llm("summarizer", model_name=getattr(config, "summarizer_model", None), temperature=0.0)
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt),
    ]

    summary_text = None
    failure_err = None

    for attempt in range(config.max_retries):
        try:
            resp = llm.invoke(messages)
            summary_text = resp.content.strip()
            # Enforce token / word cap truncation if needed
            words = summary_text.split()
            if len(words) > token_cap:
                summary_text = " ".join(words[:token_cap]) + "..."
            break
        except Exception as e:
            failure_err = e
            logger.warning("Summarizer attempt %d/%d failed: %s", attempt + 1, config.max_retries, e)

    if not summary_text:
        err_msg = f"Summarizer failed after {config.max_retries} attempts: {failure_err}"
        logger.error(err_msg)
        return {
            "error": err_msg,
            "recorded_failures": [{"agent": "summarizer", "round": current_round, "error": str(failure_err)}],
            "audit_trail": [format_audit_log("Summarizer", f"FAILED: {err_msg}")],
        }

    log_msg = format_audit_log(
        "Summarizer",
        f"Round {current_round} debate summary generated ({len(summary_text.split())} words): {summary_text[:90]}..."
    )

    return {
        "fcmad_debate_summary": summary_text,
        "audit_trail": [log_msg],
        "error": None,
    }
