import os
import sys
import json
import time
import re
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from config import config
from models.verdict import VerdictType
from tools.evidence import evidence_pool
from runner import run_audit
from utils.llm import get_llm
from utils.metrics import metrics
from eval.calibration import Calibrator, calibrate_dev_and_test, label_to_idx, CLASS_NAMES
from eval.eval_metrics import evaluate_run_records, aggregate_seeds_metrics
from eval.loaders import load_smoke_test, split_dev_test

logger = logging.getLogger(__name__)


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Computes dollar cost based on configured rates per 1M tokens."""
    p_rate = getattr(config, "prompt_cost_per_million", 1.0)
    c_rate = getattr(config, "completion_cost_per_million", 3.0)
    return round((prompt_tokens * p_rate + completion_tokens * c_rate) / 1_000_000, 6)


# ── Condition 1: Single-Agent Chain-of-Thought (CoT) ──────────────────────────

def run_single_claim_cot(
    claim_text: str,
    evidence: Optional[List[Any]] = None,
    backbone_model: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes single-agent Chain-of-Thought (CoT) verification with step-by-step reasoning.
    Uses the exact same evidence and backbone model as other conditions.
    """
    model_name = backbone_model or getattr(config, "eval_backbone_model", "claude-haiku-4-5-20251001")
    llm = get_llm("judge", model_name=model_name, temperature=0.0, seed=seed)

    evidence_str = ""
    if evidence:
        evidence_str = "\n".join(
            f"[{e.get('evidence_id', f'E{i+1}')}]: {e.get('snippet', e.get('text', ''))}"
            if isinstance(e, dict) else str(e)
            for i, e in enumerate(evidence)
        )
    else:
        evidence_str = "No external evidence provided. Rely on factual reasoning."

    prompt = f"""You are a rigorous fact-checking assistant using Chain-of-Thought reasoning.
Evaluate the factual truthfulness of the target claim against the provided evidence.

Target Claim:
"{claim_text}"

Available Evidence:
{evidence_str}

Reason step-by-step through the evidence. Conclude with a 3-class probability distribution
and authoritative verdict (SUPPORTED, REFUTED, or NOT_ENOUGH_INFORMATION).

Respond in JSON format ONLY:
{{
  "reasoning": "<step-by-step reasoning>",
  "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFORMATION",
  "probabilities": {{"supported": 0.85, "refuted": 0.10, "unknown": 0.05}}
}}"""

    from langchain_core.messages import HumanMessage
    last_error = None
    t0 = time.time()
    p_tokens = 0
    c_tokens = 0

    for attempt in range(config.max_retries):
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                p_tokens += resp.usage_metadata.get("input_tokens", 0)
                c_tokens += resp.usage_metadata.get("output_tokens", 0)

            raw = str(getattr(resp, "content", "")).strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            raw_v = str(data.get("verdict", "")).strip().upper()
            if raw_v not in CLASS_NAMES:
                raise ValueError(f"Invalid verdict '{raw_v}'")

            probs_dict = data.get("probabilities", {})
            p_s = float(probs_dict.get("supported", 0.33))
            p_r = float(probs_dict.get("refuted", 0.33))
            p_n = float(probs_dict.get("unknown", probs_dict.get("not_enough_information", 0.34)))
            total_p = p_s + p_r + p_n
            if total_p > 0:
                dist = [round(p_s / total_p, 4), round(p_r / total_p, 4), round(p_n / total_p, 4)]
            else:
                dist = [1.0 / 3, 1.0 / 3, 1.0 / 3]

            conf = float(np.max(dist))
            latency = round(time.time() - t0, 3)
            cost = compute_cost(p_tokens, c_tokens)

            return {
                "predicted_label": raw_v,
                "probabilities": dist,
                "final_distribution": dist,
                "raw_confidence": conf,
                "calibrated_confidence": conf,
                "abstain": False,
                "failed": False,
                "error": None,
                "latency": latency,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "total_tokens": p_tokens + c_tokens,
                "cost_usd": cost,
                "total_llm_calls": attempt + 1,
            }
        except Exception as e:
            last_error = e

    latency = round(time.time() - t0, 3)
    cost = compute_cost(p_tokens, c_tokens)
    return {
        "predicted_label": None,
        "probabilities": None,
        "final_distribution": None,
        "raw_confidence": None,
        "calibrated_confidence": None,
        "abstain": True,
        "failed": True,
        "error": str(last_error),
        "latency": latency,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "total_tokens": p_tokens + c_tokens,
        "cost_usd": cost,
        "total_llm_calls": config.max_retries,
    }


# ── Condition 2: Self-Consistency (SC) ────────────────────────────────────────

def run_single_claim_sc(
    claim_text: str,
    evidence: Optional[List[Any]] = None,
    num_samples: int = 5,
    backbone_model: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes Self-Consistency (SC) sampling with the exact same total LLM calls
    as FC-MAD. Samples stochastic CoT reasoning at temperature > 0 and pools distributions.
    """
    model_name = backbone_model or getattr(config, "eval_backbone_model", "claude-haiku-4-5-20251001")
    temp = getattr(config, "debater_temperature", 0.7)
    k_samples = max(1, num_samples)

    distributions: List[List[float]] = []
    total_p_tokens = 0
    total_c_tokens = 0
    t0 = time.time()
    failures = 0

    for i in range(k_samples):
        sample_seed = seed * 1000 + i
        res = run_single_claim_cot(
            claim_text=claim_text,
            evidence=evidence,
            backbone_model=model_name,
            seed=sample_seed,
        )
        total_p_tokens += res.get("prompt_tokens", 0)
        total_c_tokens += res.get("completion_tokens", 0)

        if res.get("failed") or not res.get("probabilities"):
            failures += 1
        else:
            distributions.append(res["probabilities"])

    latency = round(time.time() - t0, 3)
    cost = compute_cost(total_p_tokens, total_c_tokens)

    if not distributions:
        return {
            "predicted_label": None,
            "probabilities": None,
            "final_distribution": None,
            "raw_confidence": None,
            "calibrated_confidence": None,
            "abstain": True,
            "failed": True,
            "error": f"All {k_samples} self-consistency samples failed.",
            "latency": latency,
            "prompt_tokens": total_p_tokens,
            "completion_tokens": total_c_tokens,
            "total_tokens": total_p_tokens + total_c_tokens,
            "cost_usd": cost,
            "total_llm_calls": k_samples,
        }

    # Average probability distributions across stochastic samples
    avg_dist = np.mean(distributions, axis=0)
    avg_dist = [float(round(v, 4)) for v in avg_dist]
    pred_idx = int(np.argmax(avg_dist))
    pred_label = CLASS_NAMES[pred_idx]
    conf = float(np.max(avg_dist))

    return {
        "predicted_label": pred_label,
        "probabilities": avg_dist,
        "final_distribution": avg_dist,
        "raw_confidence": conf,
        "calibrated_confidence": conf,
        "abstain": False,
        "failed": False,
        "error": None,
        "latency": latency,
        "prompt_tokens": total_p_tokens,
        "completion_tokens": total_c_tokens,
        "total_tokens": total_p_tokens + total_c_tokens,
        "cost_usd": cost,
        "total_llm_calls": k_samples,
    }


# ── Condition 3 & 4: Veritas Mode & FC-MAD Mode (and Ablations) ───────────────

def run_single_claim_audit(
    claim_text: str,
    mode: str = "fcmad",
    evidence: Optional[List[Any]] = None,
    max_rounds: Optional[int] = None,
    ablation: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a verification audit via runner.py (supporting veritas, fcmad, and fcmad ablations).
    Extracts telemetry (tokens, latency, cost), distributions, and verdicts.
    """
    t0 = time.time()
    try:
        audit_res = run_audit(
            text=claim_text,
            mode=mode,
            max_rounds=max_rounds,
            evidence=evidence,
            ablation=ablation,
        )

        p_tokens = metrics.total_prompt_tokens
        c_tokens = metrics.total_completion_tokens
        tot_calls = metrics.total_llm_calls
        latency = audit_res.execution_time_seconds or round(time.time() - t0, 3)
        cost = compute_cost(p_tokens, c_tokens)

        verdict_obj = audit_res.verdicts[0] if audit_res.verdicts else None
        if not verdict_obj:
            raise RuntimeError("No verdicts returned by audit session.")

        v_str = verdict_obj.verdict.value if hasattr(verdict_obj.verdict, "value") else str(verdict_obj.verdict)
        final_dist = verdict_obj.final_distribution or [1.0 / 3, 1.0 / 3, 1.0 / 3]
        raw_conf = verdict_obj.raw_confidence if verdict_obj.raw_confidence is not None else float(verdict_obj.confidence)
        cal_conf = verdict_obj.calibrated_confidence if verdict_obj.calibrated_confidence is not None else raw_conf

        return {
            "predicted_label": v_str,
            "probabilities": final_dist,
            "final_distribution": final_dist,
            "raw_confidence": raw_conf,
            "calibrated_confidence": cal_conf,
            "abstain": bool(verdict_obj.abstain),
            "failed": False,
            "error": None,
            "latency": latency,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
            "cost_usd": cost,
            "total_llm_calls": tot_calls,
        }
    except Exception as e:
        latency = round(time.time() - t0, 3)
        p_tokens = metrics.total_prompt_tokens
        c_tokens = metrics.total_completion_tokens
        return {
            "predicted_label": None,
            "probabilities": None,
            "final_distribution": None,
            "raw_confidence": None,
            "calibrated_confidence": None,
            "abstain": True,
            "failed": True,
            "error": str(e),
            "latency": latency,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
            "cost_usd": compute_cost(p_tokens, c_tokens),
            "total_llm_calls": metrics.total_llm_calls,
        }


# ── Full Evaluation Harness ───────────────────────────────────────────────────

class EvaluationHarness:
    """
    Multi-condition benchmark runner supporting:
    - Conditions: cot, sc (call-matched to fcmad), veritas, fcmad
    - Ablations: fcmad_no_judge, fcmad_no_degroot, fcmad_no_summarizer
    - DEV-split calibration fitting and separate TEST evaluation
    - Resumable runs with disk persistence
    - Mock-LLM dry-run mode
    - Output formats: JSON and formatted Markdown comparison table
    """
    def __init__(
        self,
        output_dir: str = "eval_results",
        backbone_model: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.output_dir = output_dir
        self.backbone_model = backbone_model or getattr(config, "eval_backbone_model", "claude-haiku-4-5-20251001")
        self.dry_run = dry_run
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_checkpoint_path(self, run_id: str) -> str:
        return os.path.join(self.output_dir, f"checkpoint_{run_id}.json")

    def _load_checkpoint(self, run_id: str) -> Dict[str, Any]:
        cp_path = self._get_checkpoint_path(run_id)
        if os.path.exists(cp_path):
            try:
                with open(cp_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read checkpoint %s: %s", cp_path, e)
        return {"completed_runs": {}}

    def _save_checkpoint(self, run_id: str, data: Dict[str, Any]):
        cp_path = self._get_checkpoint_path(run_id)
        try:
            with open(cp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Could not write checkpoint %s: %s", cp_path, e)

    def evaluate_condition_on_dataset(
        self,
        condition: str,
        dataset: List[Dict[str, Any]],
        seed: int = 42,
        call_budget_map: Optional[Dict[str, int]] = None,
        run_id: str = "default",
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a single condition across claims in dataset, using resumable checkpointing.
        """
        checkpoint = self._load_checkpoint(run_id)
        completed = checkpoint.get("completed_runs", {})
        results: List[Dict[str, Any]] = []

        for item in dataset:
            claim_id = str(item.get("id", f"claim_{len(results)}"))
            claim_text = item.get("claim", "")
            gt = item.get("ground_truth", "NOT_ENOUGH_INFORMATION")
            gold_ev = item.get("evidence")

            cp_key = f"{condition}|{seed}|{claim_id}"
            if cp_key in completed:
                cached_res = completed[cp_key]
                results.append(cached_res)
                continue

            # Execute condition
            if condition == "cot":
                res = run_single_claim_cot(claim_text, evidence=gold_ev, backbone_model=self.backbone_model, seed=seed)
            elif condition == "sc":
                budget = (call_budget_map or {}).get(claim_id, 11)  # Default 11 matching round 0 FCMAD
                res = run_single_claim_sc(claim_text, evidence=gold_ev, num_samples=budget, backbone_model=self.backbone_model, seed=seed)
            elif condition == "veritas":
                res = run_single_claim_audit(claim_text, mode="veritas", evidence=gold_ev, max_rounds=3, seed=seed)
            elif condition == "fcmad":
                res = run_single_claim_audit(claim_text, mode="fcmad", evidence=gold_ev, max_rounds=config.max_rounds_fcmad, seed=seed)
            elif condition.startswith("fcmad_"):
                abl = condition.replace("fcmad_", "")
                res = run_single_claim_audit(claim_text, mode="fcmad", evidence=gold_ev, max_rounds=config.max_rounds_fcmad, ablation=abl, seed=seed)
            else:
                raise ValueError(f"Unknown condition: '{condition}'")

            res["claim_id"] = claim_id
            res["claim_text"] = claim_text
            res["ground_truth"] = gt
            res["condition"] = condition
            res["seed"] = seed

            results.append(res)
            completed[cp_key] = res
            checkpoint["completed_runs"] = completed
            self._save_checkpoint(run_id, checkpoint)

        return results

    def run_benchmark(
        self,
        dataset: List[Dict[str, Any]],
        conditions: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        dev_ratio: float = 0.5,
        target_accuracy: float = 0.90,
        run_name: str = "benchmark",
    ) -> Dict[str, Any]:
        """
        Full scientific benchmark orchestration:
        1. Splits dataset into separate DEV and TEST partitions.
        2. Fits calibration parameters (T, tau) strictly on DEV partition using FC-MAD.
        3. Evaluates all conditions and ablations on TEST partition across seeds.
        4. Matches self-consistency LLM calls to FC-MAD total calls per claim.
        5. Aggregates multi-seed metrics (accuracy, macro-F1, ECE, Brier, coverage, tokens, latency, cost).
        6. Outputs structured JSON and formatted Markdown table.
        """
        all_conditions = conditions or [
            "cot",
            "sc",
            "fcmad",
            "fcmad_no_judge",
            "fcmad_no_degroot",
            "fcmad_no_summarizer",
        ]
        eval_seeds = seeds or getattr(config, "eval_seeds", [42, 43, 44])

        is_mock_run = self.dry_run or config.llm_provider == "mock" or (not config.anthropic_api_key and not config.openai_api_key)
        run_id = f"{run_name}_{int(time.time())}"

        # 1. Dev / Test Split
        dev_set, test_set = split_dev_test(dataset, dev_ratio=dev_ratio, seed=eval_seeds[0], stratify=True)
        if not test_set:
            test_set = dev_set  # Smoke test fallback if dataset too small to split

        # 2. Calibration: Fit on DEV only
        calib_save_path = os.path.join(self.output_dir, f"calibration_params_{run_id}.json")
        logger.info("Evaluating DEV partition (%d claims) with FC-MAD to fit calibration parameters...", len(dev_set))
        dev_fcmad_records = self.evaluate_condition_on_dataset("fcmad", dev_set, seed=eval_seeds[0], run_id=f"dev_{run_id}")
        calibrator = Calibrator(target_accuracy=target_accuracy)
        calibrator.fit_on_dev(dev_fcmad_records, target_accuracy=target_accuracy)
        calibrator.save_to_json(calib_save_path)

        # 3. Evaluate TEST partition across conditions and seeds
        condition_results: Dict[str, List[Dict[str, Any]]] = {}
        condition_summaries: Dict[str, Dict[str, Any]] = {}

        # First, run FC-MAD on TEST for seed 0 to obtain call budget per claim for Self-Consistency
        fcmad_seed0_runs = self.evaluate_condition_on_dataset("fcmad", test_set, seed=eval_seeds[0], run_id=run_id)
        call_budget_map = {r["claim_id"]: r.get("total_llm_calls", 11) for r in fcmad_seed0_runs}

        for cond in all_conditions:
            seed_runs_stats: List[Dict[str, Any]] = []
            for s in eval_seeds:
                records = self.evaluate_condition_on_dataset(
                    condition=cond,
                    dataset=test_set,
                    seed=s,
                    call_budget_map=call_budget_map if cond == "sc" else None,
                    run_id=run_id,
                )
                # Apply frozen calibration parameters to TEST outputs
                calibrated_records = calibrator.apply(records)
                run_stats = evaluate_run_records(calibrated_records)
                seed_runs_stats.append(run_stats)

            agg_stats = aggregate_seeds_metrics(seed_runs_stats)
            condition_summaries[cond] = agg_stats

        # 4. Generate Markdown Table
        table_md = self.generate_markdown_table(condition_summaries, is_mock=is_mock_run, num_test=len(test_set))

        # 5. Save JSON and Markdown artifacts
        output_payload = {
            "run_id": run_id,
            "run_name": run_name,
            "is_mock_run": is_mock_run,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_size_total": len(dataset),
            "dev_size": len(dev_set),
            "test_size": len(test_set),
            "seeds": eval_seeds,
            "calibration_parameters": {
                "temperature": calibrator.temperature,
                "abstain_threshold": calibrator.abstain_threshold,
                "target_accuracy": calibrator.target_accuracy,
                "dev_samples": calibrator.dev_samples_count,
            },
            "condition_summaries": condition_summaries,
            "markdown_table": table_md,
        }

        json_out_path = os.path.join(self.output_dir, f"{run_id}_results.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        md_out_path = os.path.join(self.output_dir, f"{run_id}_table.md")
        with open(md_out_path, "w", encoding="utf-8") as f:
            f.write(table_md)

        try:
            print("\n" + table_md + "\n")
        except UnicodeEncodeError:
            # Fallback for Windows consoles using non-UTF-8 codepages
            safe_text = table_md.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
            print("\n" + safe_text + "\n")

        print(f"Results saved to: {json_out_path}")
        print(f"Markdown table saved to: {md_out_path}")

        return output_payload

    @staticmethod
    def generate_markdown_table(
        summaries: Dict[str, Dict[str, Any]],
        is_mock: bool = False,
        num_test: int = 0,
    ) -> str:
        """Constructs an academic GitHub Flavored Markdown comparison table."""
        source_note = (
            "> **NOTICE:** Results generated in **MOCK DRY-RUN MODE** (deterministic MockLLM).\n"
            "> Token counts and structural pipeline behavior are verified; run with frontier LLMs for empirical benchmarking."
            if is_mock else
            "> **NOTICE:** Results generated with **LIVE FRONTIER MODELS** + live retrieval."
        )

        headers = [
            "Condition / Ablation",
            "Accuracy",
            "Macro F1",
            "ECE (10-bin)",
            "Brier Score",
            "Coverage",
            "Tokens / Claim",
            "Latency (s)",
            "Cost / Claim",
            "Failed Items",
        ]

        rows = []
        cond_labels = {
            "cot": "1. Single-Agent CoT",
            "sc": "2. Self-Consistency (Call-Matched)",
            "fcmad": "3. FC-MAD (Proposed Architecture)",
            "fcmad_no_judge": "   |- Ablation: Equal Weights (No Judge)",
            "fcmad_no_degroot": "   |- Ablation: No DeGroot (alpha=1.0)",
            "fcmad_no_summarizer": "   |- Ablation: No Summarizer",
        }

        for cond, stats in summaries.items():
            label = cond_labels.get(cond, cond)
            m = stats.get("metrics", {})
            acc = m.get("accuracy", {}).get("formatted", "N/A")
            f1 = m.get("macro_f1", {}).get("formatted", "N/A")
            ece = m.get("ece", {}).get("formatted", "N/A")
            brier = m.get("brier", {}).get("formatted", "N/A")
            cov = m.get("coverage", {}).get("formatted", "N/A")
            tokens = m.get("avg_tokens_per_claim", {}).get("formatted", "N/A")
            lat = m.get("avg_latency_per_claim", {}).get("formatted", "N/A")
            cost = m.get("avg_cost_per_claim", {}).get("formatted", "N/A")
            failed = str(stats.get("failed_count_total", 0))

            rows.append(f"| {label:<40} | {acc:<14} | {f1:<14} | {ece:<14} | {brier:<14} | {cov:<14} | {tokens:<16} | {lat:<14} | {cost:<15} | {failed:<12} |")

        sep_line = "| " + " | ".join(["-" * 40, "-" * 14, "-" * 14, "-" * 14, "-" * 14, "-" * 14, "-" * 16, "-" * 14, "-" * 15, "-" * 12]) + " |"
        hdr_line = "| " + " | ".join([f"{h:<{w}}" for h, w in zip(headers, [40, 14, 14, 14, 14, 14, 16, 14, 15, 12])]) + " |"

        return f"""### Empirical Evaluation & Ablation Comparison Table (TEST Partition, N={num_test})

{source_note}

{hdr_line}
{sep_line}
""" + "\n".join(rows)
