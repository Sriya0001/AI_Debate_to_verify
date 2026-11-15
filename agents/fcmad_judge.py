import json
import re
import logging
from typing import Dict, Any, List
import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.agents import AgentPosition
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log
from config import config
from consensus.fcmad_math import compute_credibilities, validate_belief

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Judge Agent in the FC-MAD verification system.
Your sole responsibility is to evaluate the argument quality of each debater:
- Grounding in provided evidence (citing evidence IDs accurately).
- Logical consistency and sound epistemic reasoning.
- Objective balance and avoidance of dogmatic or fabricated claims.

Assign an objective argument-quality score q_i in [0.0, 1.0] for each debater:
- 1.0: Rigorous, primary evidence citations with impeccable reasoning.
- 0.5: Moderate reasoning with minor gaps or secondary citations.
- 0.0: Dogmatic, unsubstantiated speculation with no evidence grounding.

Respond in JSON format ONLY:
{
  "argument_quality_scores": {
    "debater_0": 0.85,
    "debater_1": 0.80,
    "debater_2": 0.88
  },
  "rationale": "<concise explanation of quality score allocations>"
}"""


def fcmad_judge_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Evaluates argument quality scores q_i in [0, 1] per debater.
    Computes mathematical credibility c_i via consensus/fcmad_math.py (never by LLM).
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        err_msg = "FCMAD Judge: Target claim not found in state."
        return {
            "error": err_msg,
            "recorded_failures": [{"agent": "judge", "error": err_msg}],
            "audit_trail": [format_audit_log("Judge", f"Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    current_round = state.get("current_round", 0)
    evidence_text = evidence_pool.format_pool_for_prompt(claim_id)

    # Gather debater positions
    positions: List[AgentPosition] = state.get("agent_positions", [])
    claim_positions = [
        p for p in positions
        if p.claim_id == claim_id and p.agent_name.startswith("debater_")
    ]

    latest_by_debater: Dict[str, AgentPosition] = {}
    for p in claim_positions:
        latest_by_debater[p.agent_name] = p

    debater_names = sorted(latest_by_debater.keys())
    if len(debater_names) < 2:
        # Fallback to configured number of debaters
        num_d = getattr(config, "num_debaters", 3)
        debater_names = [f"debater_{i}" for i in range(num_d)]

    # Collect beliefs
    belief_vectors_dict = state.get("belief_vectors", {})
    beliefs_list: List[List[float]] = []
    for name in debater_names:
        if name in belief_vectors_dict:
            beliefs_list.append(belief_vectors_dict[name].to_list())
        elif name in latest_by_debater:
            beliefs_list.append(latest_by_debater[name].probabilities.to_list())
        else:
            beliefs_list.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    # Ablation: FC-MAD without Judge / Credibility (Equal Weights)
    if state.get("fcmad_equal_weights") or getattr(config, "fcmad_equal_weights", False):
        num_d = max(1, len(debater_names))
        eq_cred = [float(round(1.0 / num_d, 4)) for _ in range(num_d)]
        log_msg = format_audit_log(
            "Judge (FC-MAD Ablation: Equal Weights)",
            f"Bypassed Judge quality scores; assigned uniform credibilities c_i={1.0 / num_d:.3f} across {num_d} debaters."
        )
        return {
            "credibilities": eq_cred,
            "fcmad_quality_scores": {name: 1.0 for name in debater_names},
            "audit_trail": [log_msg],
            "error": None,
        }

    arguments_text = "\n".join(
        f"- {name.upper()}: Cited={latest_by_debater[name].cited_evidence_ids if name in latest_by_debater else []}\n"
        f"  Argument: {latest_by_debater[name].reasoning if name in latest_by_debater else 'No argument'}"
        for name in debater_names
    )

    user_prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Retrieved Evidence Pool:
{evidence_text}

Debater Arguments:
{arguments_text}

Evaluate argument quality scores q_i in [0.0, 1.0] for each debater: {', '.join(debater_names)}."""

    llm = get_llm("judge", temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    q_dict: Dict[str, float] = {}
    failure_err = None
    audit_entries: List[str] = []

    for attempt in range(config.max_retries):
        try:
            resp = llm.invoke(messages)
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            scores = data.get("argument_quality_scores", {})
            for name in debater_names:
                q_dict[name] = float(np.clip(float(scores.get(name, 1.0)), 0.0, 1.0))
            break
        except Exception as e:
            failure_err = e
            logger.warning("FCMAD Judge attempt %d/%d failed: %s", attempt + 1, config.max_retries, e)

    if not q_dict:
        logger.warning(
            "FCMAD Judge failed after %d retries (%s); defaulting argument quality q_i=1.0 per spec.",
            config.max_retries, failure_err
        )
        audit_entries.append(
            format_audit_log(
                "Judge",
                f"WARNING: Argument quality evaluation failed ({failure_err}); using default q_i=1.0."
            )
        )
        for name in debater_names:
            q_dict[name] = 1.0

    # Mathematical credibility computed strictly by fcmad_math.py
    q_scores_array = np.array([q_dict[name] for name in debater_names], dtype=float)
    credibilities = compute_credibilities(beliefs_list, quality_scores=q_scores_array)
    cred_list = [float(round(c, 4)) for c in credibilities]

    cred_str = ", ".join(f"{name}: {c:.3f} (q={q_dict[name]:.2f})" for name, c in zip(debater_names, cred_list))
    log_msg = format_audit_log(
        "Judge (FC-MAD)",
        f"Round {current_round} computed credibilities: [{cred_str}]"
    )
    audit_entries.append(log_msg)

    return {
        "credibilities": cred_list,
        "fcmad_quality_scores": q_dict,
        "audit_trail": audit_entries,
        "error": None,
    }
