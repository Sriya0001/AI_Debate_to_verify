import pytest
import math
import numpy as np

from eval.eval_metrics import (
    compute_macro_f1,
    compute_ece_10_bins,
    compute_brier_score,
    evaluate_run_records,
    aggregate_seeds_metrics,
)


def test_compute_macro_f1_three_classes():
    # 3 classes: 0: SUPPORTED, 1: REFUTED, 2: NOT_ENOUGH_INFORMATION
    predictions   = [0, 0, 1, 1, 2, 2]
    ground_truths = [0, 1, 1, 1, 2, 2]

    # Class 0: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5 -> F1 = 2/3
    # Class 1: TP=2, FP=1, FN=0 -> Prec=2/3, Rec=1.0 -> F1 = 4/5 = 0.80
    # Class 2: TP=2, FP=0, FN=0 -> Prec=1.0, Rec=1.0 -> F1 = 1.0
    # Macro F1 = (2/3 + 0.80 + 1.0) / 3 = (0.6667 + 0.8 + 1.0)/3 = 0.8222
    f1 = compute_macro_f1(predictions, ground_truths, num_classes=3)
    assert abs(f1 - 0.8222) < 1e-3


def test_compute_ece_10_bins_perfect_calibration():
    # Perfectly calibrated predictions: confidence 1.0 always correct, 0.5 half correct
    confidences   = [1.0, 1.0, 0.5, 0.5]
    predictions   = [0,   1,   0,   0  ]
    ground_truths = [0,   1,   0,   1  ]

    ece = compute_ece_10_bins(confidences, predictions, ground_truths, num_bins=10)
    assert abs(ece - 0.0) < 1e-4


def test_compute_brier_score_bounds():
    # Perfect predictions have Brier score 0
    distributions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    ground_truths = [0, 1]
    assert compute_brier_score(distributions, ground_truths) == 0.0

    # Completely wrong predictions
    wrong_distributions = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    # (0 - 1)^2 + (1 - 0)^2 = 2 per sample
    assert abs(compute_brier_score(wrong_distributions, ground_truths) - 2.0) < 1e-4


def test_evaluate_run_records_failure_and_abstain_handling():
    records = [
        # Correct, non-abstained
        {"claim_id": "C1", "predicted_label": "SUPPORTED", "ground_truth": "SUPPORTED", "raw_confidence": 0.90, "calibrated_confidence": 0.90, "final_distribution": [0.90, 0.05, 0.05], "abstain": False, "failed": False, "latency": 1.2, "prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.00025},
        # Wrong, non-abstained
        {"claim_id": "C2", "predicted_label": "REFUTED", "ground_truth": "SUPPORTED", "raw_confidence": 0.80, "calibrated_confidence": 0.80, "final_distribution": [0.10, 0.80, 0.10], "abstain": False, "failed": False, "latency": 1.0, "prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.00025},
        # Abstained (confidence too low), should be excluded from accuracy & macro-F1, counted in coverage
        {"claim_id": "C3", "predicted_label": "NOT_ENOUGH_INFORMATION", "ground_truth": "REFUTED", "raw_confidence": 0.40, "calibrated_confidence": 0.40, "final_distribution": [0.30, 0.30, 0.40], "abstain": True, "failed": False, "latency": 1.1, "prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.00025},
        # Failed (LLM error or parse failure), must be excluded from all metrics and counted separately
        {"claim_id": "C4", "predicted_label": None, "ground_truth": "SUPPORTED", "raw_confidence": None, "calibrated_confidence": None, "final_distribution": None, "abstain": True, "failed": True, "latency": 0.5, "prompt_tokens": 50, "completion_tokens": 0, "cost_usd": 0.00005},
    ]

    metrics = evaluate_run_records(records)

    assert metrics["total_items"] == 4
    assert metrics["failed_count"] == 1
    assert metrics["total_valid"] == 3
    assert metrics["non_abstained_count"] == 2
    # Coverage: 2 non-abstained / 3 valid = 0.6667
    assert abs(metrics["coverage"] - 0.6667) < 1e-3
    # Accuracy on non-abstained: 1 correct out of 2 = 0.50
    assert abs(metrics["accuracy"] - 0.50) < 1e-3

    # Cost, latency, and tokens are tracked per claim
    assert metrics["avg_tokens_per_claim"] > 0
    assert metrics["avg_latency_per_claim"] > 0
    assert metrics["avg_cost_per_claim"] > 0


def test_aggregate_seeds_metrics_mean_and_standard_error():
    seed_runs = [
        {"accuracy": 0.80, "macro_f1": 0.78, "ece": 0.12, "brier": 0.25, "coverage": 0.90, "avg_tokens_per_claim": 500, "avg_latency_per_claim": 1.5, "avg_cost_per_claim": 0.001, "failed_count": 0},
        {"accuracy": 0.84, "macro_f1": 0.82, "ece": 0.10, "brier": 0.22, "coverage": 0.92, "avg_tokens_per_claim": 520, "avg_latency_per_claim": 1.6, "avg_cost_per_claim": 0.0011, "failed_count": 0},
        {"accuracy": 0.82, "macro_f1": 0.80, "ece": 0.11, "brier": 0.24, "coverage": 0.91, "avg_tokens_per_claim": 510, "avg_latency_per_claim": 1.4, "avg_cost_per_claim": 0.00105, "failed_count": 1},
    ]

    agg = aggregate_seeds_metrics(seed_runs)

    assert agg["num_seeds"] == 3
    assert agg["failed_count_total"] == 1

    acc_stat = agg["metrics"]["accuracy"]
    assert abs(acc_stat["mean"] - 0.82) < 1e-4
    # Sample std of [0.80, 0.84, 0.82] = 0.02. SE = 0.02 / sqrt(3) ~= 0.0115
    assert abs(acc_stat["se"] - (0.02 / math.sqrt(3))) < 1e-3
    assert "±" in acc_stat["formatted"]
