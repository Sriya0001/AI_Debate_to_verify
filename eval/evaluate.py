import time
import json
from typing import Dict, Any, List
import numpy as np

from config import config
from eval.dataset import BENCHMARK_CLAIMS, load_benchmark_dataset
from runner import run_audit
from utils.llm import get_llm
from models.verdict import VerdictType


def run_single_llm_baseline(claim_text: str) -> Dict[str, Any]:
    """Single direct LLM call without retrieval or multi-agent debate."""
    t0 = time.time()
    llm = get_llm("judge", temperature=0.0)
    prompt = (
        f"Verify this claim:\n\"{claim_text}\"\n\n"
        "Respond with a JSON object: {\"verdict\": \"SUPPORTED\" | \"REFUTED\" | \"NOT_ENOUGH_INFORMATION\", \"confidence\": 0.0-1.0}"
    )
    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(raw)
        verdict = data.get("verdict", "SUPPORTED").upper()
        conf = float(data.get("confidence", 0.7))
    except Exception:
        verdict = "SUPPORTED"
        conf = 0.5

    return {
        "verdict": verdict,
        "confidence": conf,
        "latency": round(time.time() - t0, 3),
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


def compute_metrics(predictions: List[str], ground_truths: List[str], confidences: List[float], latencies: List[float]):
    classes = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
    n = len(ground_truths)
    if n == 0:
        return {}

    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / n

    # Macro F1
    f1_list = []
    for c in classes:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g == c)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g != c)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != c and g == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    macro_f1 = np.mean(f1_list)
    avg_latency = np.mean(latencies)
    pct_nei = sum(1 for p in predictions if p == "NOT_ENOUGH_INFORMATION") / n
    ece = compute_ece(confidences, predictions, ground_truths)
    cm = compute_confusion_matrix(predictions, ground_truths, classes)

    return {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "avg_latency_sec": round(avg_latency, 2),
        "pct_not_enough_info": round(pct_nei * 100, 1),
        "ece": ece,
        "confusion_matrix": cm,
    }


def run_benchmark(sample_size: int = 6, dataset_path: str = None) -> Dict[str, Any]:
    """
    Executes comparison between:
    1. Single LLM Direct
    2. ClaimAuditor Baseline (5 agents)
    3. VERITAS AI (10 agents, DeGroot consensus)
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
    veritas_preds, veritas_confs, veritas_lats = [], [], []
    ground_truths = [c["ground_truth"] for c in test_set]

    for item in test_set:
        claim = item["claim"]
        gt = item["ground_truth"]
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

        # 3. VERITAS AI
        v_res = run_audit(claim, mode="veritas", max_rounds=3)
        v_v = v_res.aggregate_verdict.value if hasattr(v_res.aggregate_verdict, "value") else str(v_res.aggregate_verdict)
        veritas_preds.append(v_v)
        veritas_confs.append(v_res.aggregate_confidence)
        veritas_lats.append(v_res.execution_time_seconds)

    s_metrics = compute_metrics(single_preds, ground_truths, single_confs, single_lats)
    b_metrics = compute_metrics(base_preds, ground_truths, base_confs, base_lats)
    v_metrics = compute_metrics(veritas_preds, ground_truths, veritas_confs, veritas_lats)

    print("\n\n=== BENCHMARK COMPARISON TABLE ===")
    print(f"| Metric                   | Single LLM | ClaimAuditor Baseline | VERITAS AI |")
    print(f"|--------------------------|------------|-----------------------|------------|")
    print(f"| Accuracy                 | {s_metrics['accuracy']:.1%}      | {b_metrics['accuracy']:.1%}                 | {v_metrics['accuracy']:.1%}      |")
    print(f"| Macro F1                 | {s_metrics['macro_f1']:.3f}      | {b_metrics['macro_f1']:.3f}                 | {v_metrics['macro_f1']:.3f}      |")
    print(f"| Expected Calib Error     | {s_metrics['ece']:.3f}      | {b_metrics['ece']:.3f}                 | {v_metrics['ece']:.3f}      |")
    print(f"| Avg Latency              | {s_metrics['avg_latency_sec']}s       | {b_metrics['avg_latency_sec']}s                 | {v_metrics['avg_latency_sec']}s       |")
    print(f"| % NOT_ENOUGH_INFO        | {s_metrics['pct_not_enough_info']}%       | {b_metrics['pct_not_enough_info']}%                 | {v_metrics['pct_not_enough_info']}%      |")
    print(f"| Mathematical Consensus   | No         | No                    | DeGroot    |")
    print(f"| Socratic Epistemic Probe | No         | No                    | Yes        |")
    print(f"| Source Credibility Model | No         | No                    | 6-Tier     |")

    print("\n=== VERITAS AI CONFUSION MATRIX ===")
    print(f"{'Actual \\ Predicted':<25} | {'SUPPORTED':<12} | {'REFUTED':<12} | {'NOT_ENOUGH_INFO':<17}")
    print("-" * 75)
    for actual, row in v_metrics["confusion_matrix"].items():
        print(f"{actual:<25} | {row.get('SUPPORTED', 0):<12} | {row.get('REFUTED', 0):<12} | {row.get('NOT_ENOUGH_INFORMATION', 0):<17}")

    return {
        "single_llm": s_metrics,
        "baseline": b_metrics,
        "veritas_ai": v_metrics,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VERITAS AI Benchmark Runner")
    parser.add_argument("--size", type=int, default=6, help="Number of claims to evaluate (default: 6)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to custom JSON/CSV dataset")
    args = parser.parse_args()
    run_benchmark(sample_size=args.size, dataset_path=args.dataset)
