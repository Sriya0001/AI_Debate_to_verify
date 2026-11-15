import time
import json
import re
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from config import config
from eval.dataset import BENCHMARK_CLAIMS, load_benchmark_dataset
from runner import run_audit
from utils.llm import get_llm
from models.verdict import VerdictType

logger = logging.getLogger(__name__)


def run_single_llm_baseline(claim_text: str) -> Dict[str, Any]:
    """Single direct LLM call without retrieval or multi-agent debate."""
    t0 = time.time()
    llm = get_llm("judge", temperature=0.0)
    prompt = (
        f"Verify this claim:\n\"{claim_text}\"\n\n"
        "Respond with a JSON object: {\"verdict\": \"SUPPORTED\" | \"REFUTED\" | \"NOT_ENOUGH_INFORMATION\", \"confidence\": 0.0-1.0}"
    )
    from langchain_core.messages import HumanMessage

    last_error = None
    for attempt in range(config.max_retries):
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            raw_verdict = data.get("verdict")
            if not raw_verdict:
                raise ValueError("Missing 'verdict' key in LLM response.")
            verdict = str(raw_verdict).strip().upper()
            if verdict not in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]:
                raise ValueError(f"Invalid verdict '{verdict}' from LLM.")
            conf = float(data.get("confidence", 0.7))
            return {
                "verdict": verdict,
                "confidence": conf,
                "latency": round(time.time() - t0, 3),
                "failed": False,
                "error": None,
            }
        except Exception as e:
            last_error = e
            logger.warning(
                "Single LLM baseline attempt %d/%d failed: %s",
                attempt + 1,
                config.max_retries,
                e,
            )

    # Explicit failure recording: never hide an error as a prediction/verdict
    logger.error(
        "Single LLM baseline failed after %d attempt(s): %s",
        config.max_retries,
        last_error,
    )
    return {
        "verdict": None,
        "confidence": None,
        "latency": round(time.time() - t0, 3),
        "failed": True,
        "error": str(last_error),
    }


def compute_confusion_matrix(predictions: List[str], ground_truths: List[str], classes: List[str]) -> Dict[str, Dict[str, int]]:
    """Generates a NxN confusion matrix mapping (Actual, Predicted) -> Count."""
    cm = {gt: {pred: 0 for pred in classes} for gt in classes}
    for p, g in zip(predictions, ground_truths):
        if g in cm and p in cm[g]:
            cm[g][p] += 1
        elif g in cm:
            cm[g].setdefault(p, 0)
            cm[g][p] += 1
    return cm


def compute_ece(confidences: List[float], predictions: List[str], ground_truths: List[str], num_bins: int = 5) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    if not confidences or len(confidences) != len(predictions):
        return 0.0
    
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        indices = [
            idx for idx, c in enumerate(confidences)
            if (bin_lower <= c < bin_upper) or (i == num_bins - 1 and bin_lower <= c <= bin_upper)
        ]

        if not indices:
            continue

        bin_conf = np.mean([confidences[idx] for idx in indices])
        bin_acc = np.mean([1.0 if predictions[idx] == ground_truths[idx] else 0.0 for idx in indices])
        bin_weight = len(indices) / n

        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(round(ece, 3))


def compute_metrics(
    predictions: List[Optional[str]],
    ground_truths: List[str],
    confidences: List[Optional[float]],
    latencies: List[float],
    failed_count: Optional[int] = None,
) -> Dict[str, Any]:
    classes = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]

    # Filter out failed items where prediction is missing or invalid
    valid_data = [
        (p, g, c, l)
        for p, g, c, l in zip(predictions, ground_truths, confidences, latencies)
        if p is not None and p in classes and c is not None
    ]

    inferred_failed = len(predictions) - len(valid_data)
    total_failed = (failed_count if failed_count is not None else 0) + inferred_failed

    n = len(valid_data)
    if n == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "avg_latency_sec": 0.0,
            "pct_not_enough_info": 0.0,
            "ece": 0.0,
            "confusion_matrix": {gt: {pred: 0 for pred in classes} for gt in classes},
            "failed_count": total_failed,
            "total_evaluated": 0,
        }

    valid_preds = [x[0] for x in valid_data]
    valid_gts = [x[1] for x in valid_data]
    valid_confs = [x[2] for x in valid_data]
    valid_lats = [x[3] for x in valid_data]

    correct = sum(1 for p, g in zip(valid_preds, valid_gts) if p == g)
    accuracy = correct / n

    # Macro F1
    f1_list = []
    for c in classes:
        tp = sum(1 for p, g in zip(valid_preds, valid_gts) if p == c and g == c)
        fp = sum(1 for p, g in zip(valid_preds, valid_gts) if p == c and g != c)
        fn = sum(1 for p, g in zip(valid_preds, valid_gts) if p != c and g == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))
    avg_latency = float(np.mean(valid_lats))
    pct_nei = sum(1 for p in valid_preds if p == "NOT_ENOUGH_INFORMATION") / n
    ece = compute_ece(valid_confs, valid_preds, valid_gts)
    cm = compute_confusion_matrix(valid_preds, valid_gts, classes)

    return {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "avg_latency_sec": round(avg_latency, 2),
        "pct_not_enough_info": round(pct_nei * 100, 1),
        "ece": ece,
        "confusion_matrix": cm,
        "failed_count": total_failed,
        "total_evaluated": n,
    }


def run_benchmark(sample_size: int = 6, dataset_path: str = None) -> Dict[str, Any]:
    """
    Executes comparison between:
    1. Single LLM Direct
    2. ClaimAuditor Baseline (5 agents)
    3. FC-MAD (Fast Credibility-Weighted Multi-Agent Debate)
    4. VERITAS AI (10 agents, DeGroot consensus)
    """
    raw_dataset = load_benchmark_dataset(dataset_path)
    test_set = raw_dataset[:sample_size]

    is_mock = config.llm_provider == "mock" or (not config.anthropic_api_key and not config.openai_api_key)

    print(f"\n=======================================================")
    print(f"  RUNNING EMPIRICAL EVALUATION ACROSS {len(test_set)} CLAIMS")
    print(f"  LLM Provider: {config.llm_provider.upper()}")
    if is_mock:
        print(f"  [NOTE] Running with deterministic MockLLM.")
        print(f"  High/100% accuracy here verifies structural pipeline integration,")
        print(f"  LangGraph agent state loops, and DeGroot matrix math.")
        print(f"  Live open-domain benchmarks (FEVER/PolitiFact) realistically yield ~76-86%.")
    else:
        print(f"  Running in LIVE mode with real frontier LLM + web retrieval.")
    print(f"=======================================================\n")

    single_preds, single_confs, single_lats = [], [], []
    base_preds, base_confs, base_lats = [], [], []
    fcmad_preds, fcmad_confs, fcmad_lats = [], [], []
    veritas_preds, veritas_confs, veritas_lats = [], [], []
    ground_truths = [c["ground_truth"] for c in test_set]

    for item in test_set:
        claim = item["claim"]
        gt = item["ground_truth"]
        evidence = item.get("evidence")
        print(f"Evaluating [{item['id']}]: \"{claim[:55]}...\" | Truth: {gt}")

        # 1. Single LLM
        s_res = run_single_llm_baseline(claim)
        single_preds.append(s_res["verdict"])
        single_confs.append(s_res["confidence"])
        single_lats.append(s_res["latency"])

        # 2. Baseline ClaimAuditor
        b_res = run_audit(claim, mode="baseline", max_rounds=2)
        base_v = b_res.aggregate_verdict.value if hasattr(b_res.aggregate_verdict, "value") else str(b_res.aggregate_verdict)
        base_preds.append(base_v)
        base_confs.append(b_res.aggregate_confidence)
        base_lats.append(b_res.execution_time_seconds)

        # 3. FC-MAD
        f_res = run_audit(claim, mode="fcmad", max_rounds=config.max_rounds_fcmad, evidence=evidence)
        f_v = f_res.aggregate_verdict.value if hasattr(f_res.aggregate_verdict, "value") else str(f_res.aggregate_verdict)
        fcmad_preds.append(f_v)
        fcmad_confs.append(f_res.aggregate_confidence)
        fcmad_lats.append(f_res.execution_time_seconds)

        # 4. VERITAS AI
        v_res = run_audit(claim, mode="veritas", max_rounds=3, evidence=evidence)
        v_v = v_res.aggregate_verdict.value if hasattr(v_res.aggregate_verdict, "value") else str(v_res.aggregate_verdict)
        veritas_preds.append(v_v)
        veritas_confs.append(v_res.aggregate_confidence)
        veritas_lats.append(v_res.execution_time_seconds)

    s_metrics = compute_metrics(single_preds, ground_truths, single_confs, single_lats)
    b_metrics = compute_metrics(base_preds, ground_truths, base_confs, base_lats)
    f_metrics = compute_metrics(fcmad_preds, ground_truths, fcmad_confs, fcmad_lats)
    v_metrics = compute_metrics(veritas_preds, ground_truths, veritas_confs, veritas_lats)

    print("\n\n=== BENCHMARK COMPARISON TABLE ===")
    print(f"| Metric                   | Single LLM | ClaimAuditor Baseline | FC-MAD       | VERITAS AI |")
    print(f"|--------------------------|------------|-----------------------|--------------|------------|")
    print(f"| Accuracy                 | {s_metrics['accuracy']:.1%}      | {b_metrics['accuracy']:.1%}                 | {f_metrics['accuracy']:.1%}        | {v_metrics['accuracy']:.1%}      |")
    print(f"| Macro F1                 | {s_metrics['macro_f1']:.3f}      | {b_metrics['macro_f1']:.3f}                 | {f_metrics['macro_f1']:.3f}        | {v_metrics['macro_f1']:.3f}      |")
    print(f"| Expected Calib Error     | {s_metrics['ece']:.3f}      | {b_metrics['ece']:.3f}                 | {f_metrics['ece']:.3f}        | {v_metrics['ece']:.3f}      |")
    print(f"| Avg Latency              | {s_metrics['avg_latency_sec']}s       | {b_metrics['avg_latency_sec']}s                 | {f_metrics['avg_latency_sec']}s         | {v_metrics['avg_latency_sec']}s       |")
    print(f"| % NOT_ENOUGH_INFO        | {s_metrics['pct_not_enough_info']}%       | {b_metrics['pct_not_enough_info']}%                 | {f_metrics['pct_not_enough_info']}%        | {v_metrics['pct_not_enough_info']}%      |")
    print(f"| Failed Items             | {s_metrics.get('failed_count', 0)}          | {b_metrics.get('failed_count', 0)}                     | {f_metrics.get('failed_count', 0)}            | {v_metrics.get('failed_count', 0)}          |")
    print(f"| Mathematical Consensus   | No         | No                    | DeGroot TV   | DeGroot    |")
    print(f"| Socratic Epistemic Probe | No         | No                    | Summary      | Yes        |")
    print(f"| Source Credibility Model | No         | No                    | Judge Q-Score| 6-Tier     |")

    print("\n=== VERITAS AI CONFUSION MATRIX ===")
    header_col = "Actual \\ Predicted"
    print(f"{header_col:<25} | {'SUPPORTED':<12} | {'REFUTED':<12} | {'NOT_ENOUGH_INFO':<17}")
    print("-" * 75)
    for actual, row in v_metrics["confusion_matrix"].items():
        print(f"{actual:<25} | {row.get('SUPPORTED', 0):<12} | {row.get('REFUTED', 0):<12} | {row.get('NOT_ENOUGH_INFORMATION', 0):<17}")

    return {
        "single_llm": s_metrics,
        "baseline": b_metrics,
        "fcmad": f_metrics,
        "veritas_ai": v_metrics,
    }


if __name__ == "__main__":
    import argparse
    from eval.harness import EvaluationHarness
    from eval.loaders import load_smoke_test, load_fever_dataset, load_xfact_dataset

    parser = argparse.ArgumentParser(description="VERITAS AI & FC-MAD Scientific Evaluation Harness")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "fever", "xfact", "custom"], help="Dataset mode (smoke, fever, xfact, custom)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to FEVER / X-Fact / custom JSONL/CSV file")
    parser.add_argument("--size", type=int, default=None, help="Number of claims to evaluate")
    parser.add_argument("--conditions", type=str, default=None, help="Comma-separated conditions (cot,sc,veritas,fcmad,fcmad_no_judge,fcmad_no_degroot,fcmad_no_summarizer)")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated random seeds (default: 42,43,44)")
    parser.add_argument("--target-accuracy", type=float, default=0.90, help="Target accuracy for abstain threshold tau (default: 0.90)")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save JSON and Markdown results")
    parser.add_argument("--dry-run", action="store_true", help="Force Mock-LLM dry-run mode")
    parser.add_argument("--legacy", action="store_true", help="Run legacy 4-column quick comparison")
    args = parser.parse_args()

    if args.legacy:
        sample_s = args.size if args.size is not None else 6
        run_benchmark(sample_size=sample_s, dataset_path=args.dataset)
    else:
        # Load dataset
        if args.mode == "smoke":
            dataset = load_smoke_test(limit=args.size)
        elif args.mode == "fever":
            f_path = args.dataset or config.fever_dataset_path
            dataset = load_fever_dataset(f_path, max_samples=args.size, seed=42)
        elif args.mode == "xfact":
            x_path = args.dataset or config.xfact_dataset_path
            dataset = load_xfact_dataset(x_path, max_samples=args.size, seed=42)
        else:
            dataset = load_benchmark_dataset(args.dataset)
            if args.size:
                dataset = dataset[:args.size]

        conds = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
        seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

        harness = EvaluationHarness(
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        harness.run_benchmark(
            dataset=dataset,
            conditions=conds,
            seeds=seeds_list,
            target_accuracy=args.target_accuracy,
            run_name=f"{args.mode}_eval",
        )
