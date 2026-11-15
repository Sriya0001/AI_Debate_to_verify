import os
import json
import pytest
import numpy as np

from config import config
from utils.cache import LLMResponseCache
from utils.llm import get_llm, MockLLM
from eval.harness import (
    run_single_claim_cot,
    run_single_claim_sc,
    run_single_claim_audit,
    EvaluationHarness,
    compute_cost,
)


def test_sqlite_llm_cache(tmp_path):
    cache_db = str(tmp_path / "test_cache.sqlite")
    cache = LLMResponseCache(db_path=cache_db)

    # Miss
    assert cache.get("model-a", "What is the capital of France?", seed=42, temperature=0.0) is None

    # Set
    cache.set(
        model_name="model-a",
        prompt_text="What is the capital of France?",
        response_text="Paris",
        seed=42,
        temperature=0.0,
        prompt_tokens=15,
        completion_tokens=2,
        latency=0.35,
    )

    # Hit
    hit = cache.get("model-a", "What is the capital of France?", seed=42, temperature=0.0)
    assert hit is not None
    assert hit["response_text"] == "Paris"
    assert hit["prompt_tokens"] == 15
    assert hit["completion_tokens"] == 2
    assert abs(hit["latency"] - 0.35) < 1e-3

    # Different seed is a miss
    assert cache.get("model-a", "What is the capital of France?", seed=99, temperature=0.0) is None

    # Stats
    stats = cache.stats()
    assert stats["total_entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 2

    # Clear
    cache.clear()
    assert cache.stats()["total_entries"] == 0


def test_cot_condition_offline_mock():
    res = run_single_claim_cot(
        claim_text="The Eiffel Tower is in Paris.",
        evidence=[{"evidence_id": "E1", "text": "The Eiffel Tower is located in Paris, France."}],
        backbone_model="veritas-mock-engine",
        seed=42,
    )
    assert res["failed"] is False
    assert res["predicted_label"] == "SUPPORTED"
    assert res["probabilities"] is not None
    assert len(res["probabilities"]) == 3
    assert res["raw_confidence"] >= 0.70
    assert res["total_tokens"] > 0
    assert res["latency"] >= 0
    assert res["cost_usd"] >= 0


def test_self_consistency_call_budget_matching():
    k_calls = 4
    res = run_single_claim_sc(
        claim_text="Water boils at 100 degrees Celsius at sea level.",
        evidence=[{"evidence_id": "E1", "text": "Standard boiling point of water is 100 C."}],
        num_samples=k_calls,
        backbone_model="veritas-mock-engine",
        seed=42,
    )
    assert res["failed"] is False
    assert res["total_llm_calls"] == k_calls
    assert res["predicted_label"] == "SUPPORTED"
    assert len(res["final_distribution"]) == 3
    assert abs(sum(res["final_distribution"]) - 1.0) < 1e-3


def test_fcmad_ablations_offline_mock():
    # 1. Ablation: Equal weights (no judge)
    res_no_judge = run_single_claim_audit(
        claim_text="Alexander Fleming discovered penicillin in 1928.",
        mode="fcmad",
        ablation="no_judge",
        max_rounds=2,
    )
    assert res_no_judge["failed"] is False
    assert res_no_judge["predicted_label"] == "SUPPORTED"

    # 2. Ablation: No DeGroot (alpha = 1.0)
    res_no_degroot = run_single_claim_audit(
        claim_text="Alexander Fleming discovered penicillin in 1928.",
        mode="fcmad",
        ablation="no_degroot",
        max_rounds=2,
    )
    assert res_no_degroot["failed"] is False
    assert res_no_degroot["predicted_label"] == "SUPPORTED"

    # 3. Ablation: No Summarizer
    res_no_summ = run_single_claim_audit(
        claim_text="Alexander Fleming discovered penicillin in 1928.",
        mode="fcmad",
        ablation="no_summarizer",
        max_rounds=2,
    )
    assert res_no_summ["failed"] is False
    assert res_no_summ["predicted_label"] == "SUPPORTED"


def test_harness_resumable_and_dry_run(tmp_path):
    out_dir = str(tmp_path / "eval_out")
    harness = EvaluationHarness(output_dir=out_dir, dry_run=True)

    dataset = [
        {"id": "C1", "claim": "The Eiffel Tower is in Paris.", "ground_truth": "SUPPORTED", "evidence": []},
        {"id": "C2", "claim": "The Moon is made of green cheese.", "ground_truth": "REFUTED", "evidence": []},
    ]

    # Run 1: Evaluate 2 claims
    results_1 = harness.evaluate_condition_on_dataset(
        condition="cot",
        dataset=dataset,
        seed=42,
        run_id="test_run",
    )
    assert len(results_1) == 2

    # Checkpoint file created
    cp_file = harness._get_checkpoint_path("test_run")
    assert os.path.exists(cp_file)

    # Run 2: Re-run should resume from checkpoint without re-running
    results_2 = harness.evaluate_condition_on_dataset(
        condition="cot",
        dataset=dataset,
        seed=42,
        run_id="test_run",
    )
    assert len(results_2) == 2
    assert results_1[0]["claim_id"] == results_2[0]["claim_id"]


def test_harness_full_benchmark_smoke_dry_run(tmp_path):
    out_dir = str(tmp_path / "benchmark_out")
    harness = EvaluationHarness(output_dir=out_dir, dry_run=True)

    # 4 claims from smoke test
    dataset = [
        {"id": "B1", "claim": "The Eiffel Tower was built in 1889.", "ground_truth": "SUPPORTED"},
        {"id": "B2", "claim": "Water boils at 100 degrees Celsius.", "ground_truth": "SUPPORTED"},
        {"id": "B5", "claim": "The Moon is made of green cheese.", "ground_truth": "REFUTED"},
        {"id": "B9", "claim": "Extraterrestrial alien signals detected in 2026.", "ground_truth": "NOT_ENOUGH_INFORMATION"},
    ]

    benchmark_out = harness.run_benchmark(
        dataset=dataset,
        conditions=["cot", "fcmad", "fcmad_no_judge"],
        seeds=[42, 43],
        dev_ratio=0.5,
        target_accuracy=0.85,
        run_name="smoke_test_run",
    )

    assert benchmark_out["is_mock_run"] is True
    assert "condition_summaries" in benchmark_out
    assert "cot" in benchmark_out["condition_summaries"]
    assert "fcmad" in benchmark_out["condition_summaries"]
    assert "fcmad_no_judge" in benchmark_out["condition_summaries"]

    # Verify table contains clear indication of mock mode
    md_table = benchmark_out["markdown_table"]
    assert "MOCK DRY-RUN MODE" in md_table
    assert "Accuracy" in md_table
    assert "Macro F1" in md_table
    assert "ECE" in md_table
    assert "Brier" in md_table
    assert "Tokens" in md_table
    assert "Cost" in md_table
