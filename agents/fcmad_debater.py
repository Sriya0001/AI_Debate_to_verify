import json
import re
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.agents import AgentPosition, BeliefVector
from tools.evidence import evidence_pool
from utils.llm import get_llm
from utils.audit_trail import format_audit_log
from config import config
from consensus.fcmad_math import validate_belief

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are Debater {debater_id} in the FC-MAD verification system.
You are an independent, objective epistemic evaluator. You have NO assigned stance.
Your sole duty is to inspect the target claim and the provided evidence ONLY.
Do NOT fabricate facts or assume information beyond the provided evidence.

Evaluate the evidence and determine a calibrated 3-class probability distribution:
- supported: probability the claim is true and directly corroborated (0.0 to 1.0)
- refuted: probability the claim is false and directly contradicted (0.0 to 1.0)
- unknown: probability evidence is insufficient to verify or refute (0.0 to 1.0)
(All must be non-negative and sum to 1.0).

Respond in JSON format ONLY:
{{
  "position": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFORMATION",
  "probabilities": {{
    "supported": 0.70,
    "refuted": 0.15,
    "unknown": 0.15
  }},
  "cited_evidence_ids": ["E1"],
  "reasoning": "<short evidence-cited rationale citing specific evidence IDs>"
}}"""


def create_debater_node(debater_id: int, model_override: Optional[str] = None):
    """
    Factory creating a Debater agent node parameterized by id and optional model override.
    """
    agent_name = f"debater_{debater_id}"

    def debater_node(state: VeritasState) -> Dict[str, Any]:
        current_claim = state.get("current_claim")
        if not current_claim:
            err_msg = f"{agent_name}: Target claim not found in state."
            return {
                "error": err_msg,
                "recorded_failures": [{"agent": agent_name, "error": err_msg}],
                "audit_trail": [format_audit_log(agent_name, f"Failed: Target claim missing")],
            }

        claim_id = current_claim.claim_id
        evidence_text = evidence_pool.format_pool_for_prompt(claim_id)
        current_round = state.get("current_round", 0)
        summary = state.get("fcmad_debate_summary")

        # Prompt construction: Round 0 vs. Later rounds
        sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(debater_id=debater_id)
        if current_round == 0 or not summary:
            user_prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Provided Evidence Pool:
{evidence_text}

Evaluate the claim using ONLY the provided evidence. Provide your 3-class probability vector and short rationale."""
        else:
            user_prompt = f"""Target Claim [{claim_id}]:
\"{current_claim.claim_text}\"

Provided Evidence Pool:
{evidence_text}

Summary of Previous Debate Arguments:
{summary}

Re-reason from the claim, evidence, and the debate summary. Provide your revised 3-class probability vector and rationale."""

        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt),
        ]

        models_list = state.get("debater_models") or getattr(config, "debater_models", [])
        chosen_model = model_override or (models_list[debater_id] if debater_id < len(models_list) else getattr(config, "debater_model", "claude-haiku-4-5-20251001"))
        llm = get_llm("debater", model_name=chosen_model, temperature=config.debater_temperature)

        sample_vectors: List[List[float]] = []
        sample_positions: List[str] = []
        sample_cited: List[List[str]] = []
        sample_reasonings: List[str] = []
        failures: List[Dict[str, Any]] = []

        num_samples = max(1, config.samples_per_debater)

        for s_idx in range(num_samples):
            success = False
            last_err = None
            for attempt in range(config.max_retries):
                try:
                    resp = llm.invoke(messages)
                    parsed = _parse_debater_response(resp.content, agent_name, claim_id, current_round)
                    sample_vectors.append(parsed.probabilities.to_list())
                    sample_positions.append(parsed.position)
                    sample_cited.append(parsed.cited_evidence_ids)
                    sample_reasonings.append(parsed.reasoning)
                    success = True
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(
                        "%s sample %d/%d attempt %d/%d failed: %s",
                        agent_name, s_idx + 1, num_samples, attempt + 1, config.max_retries, e
                    )

            if not success:
                logger.error(
                    "%s sample %d/%d failed after %d retries: %s",
                    agent_name, s_idx + 1, num_samples, config.max_retries, last_err
                )
                failures.append({
                    "agent": agent_name,
                    "round": current_round,
                    "sample": s_idx,
                    "error": str(last_err),
                })

        # Explicit failure policy: never hide error as prediction
        if not sample_vectors:
            err_msg = f"{agent_name} failed all {num_samples} samples; no valid belief produced."
            logger.error(err_msg)
            return {
                "error": err_msg,
                "recorded_failures": failures,
                "audit_trail": [format_audit_log(agent_name, f"FAILED: {err_msg}")],
            }

        # Average samples_per_debater samples
        import numpy as np
        avg_vec = np.mean(sample_vectors, axis=0)
        avg_vec = validate_belief(avg_vec)  # Non-negative, sums to 1.0

        bv = BeliefVector(
            supported=float(round(avg_vec[0], 4)),
            refuted=float(round(avg_vec[1], 4)),
            unknown=float(round(avg_vec[2], 4)),
        )

        winning_idx = int(np.argmax(avg_vec))
        classes = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
        pos_str = classes[winning_idx]

        combined_cited = list(dict.fromkeys([cid for sublist in sample_cited for cid in sublist]))
        combined_reasoning = sample_reasonings[0] if sample_reasonings else "Evidence evaluated objectively."

        pos = AgentPosition(
            agent_name=agent_name,
            claim_id=claim_id,
            round_number=current_round,
            position=pos_str,
            probabilities=bv,
            cited_evidence_ids=combined_cited,
            reasoning=combined_reasoning,
            acknowledged_weaknesses=[],
            confidence=float(round(avg_vec[winning_idx], 4)),
            model_name=chosen_model,
        )

        log_msg = format_audit_log(
            agent_name,
            f"Round {current_round} [{chosen_model}]: P(Supp)={bv.supported:.2f}, P(Ref)={bv.refuted:.2f}, P(Unk)={bv.unknown:.2f} (sampled {len(sample_vectors)}x) | Cited: {combined_cited}"
        )

        return {
            "agent_positions": [pos],
            "belief_vectors": {agent_name: bv},
            "recorded_failures": failures,
            "audit_trail": [log_msg],
            "error": None,
        }

    return debater_node


def _parse_debater_response(raw: str, agent_name: str, claim_id: str, round_num: int) -> AgentPosition:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    probs_dict = data.get("probabilities")
    if not probs_dict or not isinstance(probs_dict, dict):
        raise ValueError(f"Missing or invalid 'probabilities' dict in {agent_name} response")

    supp = float(probs_dict.get("supported", 0.0))
    ref = float(probs_dict.get("refuted", 0.0))
    unk = float(probs_dict.get("unknown", 0.0))

    vec = validate_belief([supp, ref, unk])
    bv = BeliefVector(supported=float(vec[0]), refuted=float(vec[1]), unknown=float(vec[2]))

    pos_raw = data.get("position", "NOT_ENOUGH_INFORMATION").upper()
    if pos_raw not in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]:
        # Match argmax if unrecognized
        import numpy as np
        classes = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
        pos_raw = classes[int(np.argmax(vec))]

    return AgentPosition(
        agent_name=agent_name,
        claim_id=claim_id,
        round_number=round_num,
        position=pos_raw,
        probabilities=bv,
        cited_evidence_ids=data.get("cited_evidence_ids", []),
        reasoning=data.get("reasoning", "Evidence evaluation."),
        acknowledged_weaknesses=data.get("weaknesses", []),
        confidence=float(max(vec)),
    )
