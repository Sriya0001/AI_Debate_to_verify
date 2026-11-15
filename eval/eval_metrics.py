import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from eval.calibration import label_to_idx, CLASS_NAMES

logger = logging.getLogger(__name__)


def compute_macro_f1(predictions: List[int], ground_truths: List[int], num_classes: int = 3) -> float:
    """Computes unweighted Macro-F1 across num_classes."""
    if not predictions or len(predictions) != len(ground_truths):
        return 0.0

    f1_scores = []
    for c in range(num_classes):
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g == c)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g != c)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != c and g == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if prec + rec > 0:
            f1 = 2.0 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return float(round(float(np.mean(f1_scores)), 4))


def compute_ece_10_bins(confidences: List[float], predictions: List[int], ground_truths: List[int], num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error across 10 equal-width bins on [0, 1]
    per docs/FCMAD_MATH.md Section 8:
        ECE = sum_{b=1}^{10} (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    if not confidences or len(confidences) != len(predictions):
        return 0.0

    confs = np.asarray(confidences, dtype=float)
    preds = np.asarray(predictions, dtype=int)
    gts = np.asarray(ground_truths, dtype=int)
    n = len(confs)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for b in range(num_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        if b == num_bins - 1:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs >= low) & (confs < high)

        bin_count = np.sum(mask)
        if bin_count == 0:
            continue

        bin_conf = np.mean(confs[mask])
        bin_acc = np.mean(preds[mask] == gts[mask])
        bin_weight = bin_count / n

        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(round(float(ece), 4))


def compute_brier_score(distributions: List[List[float]], ground_truths: List[int], num_classes: int = 3) -> float:
    """
    Computes multi-class Brier score per docs/FCMAD_MATH.md Section 8:
        Brier = (1 / N) * sum_{n=1}^N sum_{k=0}^{K-1} (p_{n,k} - I(y_n == k))^2
    """
    if not distributions or len(distributions) != len(ground_truths):
        return 0.0

    n = len(distributions)
    if n == 0:
        return 0.0

    total_sq_err = 0.0
    for dist, gt in zip(distributions, ground_truths):
        for k in range(num_classes):
            target = 1.0 if k == gt else 0.0
            prob = dist[k] if k < len(dist) else 0.0
            total_sq_err += (prob - target) ** 2

    return float(round(total_sq_err / n, 4))


def evaluate_run_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes metrics for a single evaluation run (one condition, one seed).
    Strictly follows FCMAD_MATH.md:
    - Failed items excluded from metric calculation and counted separately.
    - Accuracy and Macro-F1 computed on non-abstained, non-failed claims.
    - Coverage = non-abstained / non-failed.
    - ECE (10 bins) and Brier computed on non-failed items.
    """
    failed_items = [
        r for r in records
        if r.get("failed", False) or r.get("predicted_label") is None
    ]
    failed_count = len(failed_items)

    valid_items = [
        r for r in records
        if not r.get("failed", False) and r.get("predicted_label") is not None
    ]
    total_valid = len(valid_items)

    # Telemetry aggregation (tokens, latency, cost across all attempted items)
    total_prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in records)
    total_comp_tokens = sum(int(r.get("completion_tokens", 0)) for r in records)
    total_tokens = total_prompt_tokens + total_comp_tokens
    total_latency = sum(float(r.get("latency", 0.0)) for r in records)
    total_cost = sum(float(r.get("cost_usd", 0.0)) for r in records)

    num_records = len(records)
    avg_tokens = round(total_tokens / num_records, 1) if num_records > 0 else 0.0
    avg_latency = round(total_latency / num_records, 3) if num_records > 0 else 0.0
    avg_cost = round(total_cost / num_records, 5) if num_records > 0 else 0.0

    if total_valid == 0:
        return {
            "total_items": num_records,
            "failed_count": failed_count,
            "total_evaluated": 0,
            "coverage": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "avg_tokens_per_claim": avg_tokens,
            "avg_latency_per_claim": avg_latency,
            "avg_cost_per_claim": avg_cost,
            "total_cost_usd": round(total_cost, 4),
        }

    # Abstain partitioning
    non_abstained = [r for r in valid_items if not r.get("abstain", False)]
    coverage = float(round(len(non_abstained) / total_valid, 4))

    # Accuracy and Macro-F1 on non-abstained claims
    if non_abstained:
        eval_preds = [label_to_idx(r["predicted_label"]) for r in non_abstained]
        eval_gts = [label_to_idx(r["ground_truth"]) for r in non_abstained]
        correct = sum(1 for p, g in zip(eval_preds, eval_gts) if p == g)
        accuracy = float(round(correct / len(non_abstained), 4))
        macro_f1 = compute_macro_f1(eval_preds, eval_gts)
    else:
        accuracy = 0.0
        macro_f1 = 0.0

    # Calibration & Brier metrics on all valid (non-failed) claims
    valid_confs = []
    valid_preds = []
    valid_gts = []
    valid_dists = []

    for r in valid_items:
        conf = r.get("calibrated_confidence")
        if conf is None:
            conf = r.get("raw_confidence", 0.0)
        valid_confs.append(float(conf))
        valid_preds.append(label_to_idx(r["predicted_label"]))
        valid_gts.append(label_to_idx(r["ground_truth"]))

        dist = r.get("calibrated_distribution") or r.get("final_distribution") or r.get("probabilities")
        if dist:
            d_arr = np.asarray(dist, dtype=float)
            if d_arr.sum() > 0:
                d_arr = d_arr / d_arr.sum()
            else:
                d_arr = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
            valid_dists.append(d_arr.tolist())
        else:
            # One-hot fallback from pred
            fallback_dist = [0.0, 0.0, 0.0]
            fallback_dist[label_to_idx(r["predicted_label"])] = 1.0
            valid_dists.append(fallback_dist)

    ece = compute_ece_10_bins(valid_confs, valid_preds, valid_gts, num_bins=10)
    brier = compute_brier_score(valid_dists, valid_gts, num_classes=3)

    return {
        "total_items": num_records,
        "failed_count": failed_count,
        "total_valid": total_valid,
        "non_abstained_count": len(non_abstained),
        "coverage": coverage,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "brier": brier,
        "avg_tokens_per_claim": avg_tokens,
        "avg_latency_per_claim": avg_latency,
        "avg_cost_per_claim": avg_cost,
        "total_cost_usd": round(total_cost, 4),
    }


def aggregate_seeds_metrics(seed_run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates metrics across multiple random seeds, computing mean +/- standard error (SE = std / sqrt(N)).
    Returns structured stats with formatted display strings.
    """
    if not seed_run_metrics:
        return {}

    num_seeds = len(seed_run_metrics)
    metric_keys = [
        "accuracy", "macro_f1", "ece", "brier", "coverage",
        "avg_tokens_per_claim", "avg_latency_per_claim", "avg_cost_per_claim"
    ]

    summary: Dict[str, Any] = {
        "num_seeds": num_seeds,
        "failed_count_total": sum(m.get("failed_count", 0) for m in seed_run_metrics),
        "metrics": {},
    }

    for key in metric_keys:
        values = [float(m.get(key, 0.0)) for m in seed_run_metrics]
        mean_val = float(np.mean(values))
        if num_seeds > 1:
            std_val = float(np.std(values, ddof=1))
            se_val = float(std_val / math.sqrt(num_seeds))
        else:
            std_val = 0.0
            se_val = 0.0

        is_pct = key in ("accuracy", "coverage")
        if is_pct:
            formatted = f"{mean_val * 100:.1f}% ± {se_val * 100:.1f}%"
        elif key in ("avg_tokens_per_claim", "avg_latency_per_claim"):
            formatted = f"{mean_val:.2f} ± {se_val:.2f}"
        elif key == "avg_cost_per_claim":
            formatted = f"${mean_val:.4f} ± ${se_val:.4f}"
        else:
            formatted = f"{mean_val:.3f} ± {se_val:.3f}"

        summary["metrics"][key] = {
            "mean": round(mean_val, 4),
            "se": round(se_val, 4),
            "formatted": formatted,
        }

    return summary
