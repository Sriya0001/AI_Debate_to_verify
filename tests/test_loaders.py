import json
import pytest
import os

from eval.loaders import (
    load_smoke_test,
    load_fever_dataset,
    split_dev_test,
    stratified_sample,
)


def test_load_smoke_test():
    claims = load_smoke_test()
    assert len(claims) == 12
    # Verify required keys
    for c in claims:
        assert "id" in c
        assert "claim" in c
        assert "ground_truth" in c
        assert c["ground_truth"] in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]

    limited = load_smoke_test(limit=3)
    assert len(limited) == 3


def test_load_fever_dataset_file_not_found():
    with pytest.raises(FileNotFoundError) as exc_info:
        load_fever_dataset("non_existent_fever_file_12345.jsonl")
    assert "not found" in str(exc_info.value)


def test_load_fever_dataset_valid_file(tmp_path):
    fever_file = tmp_path / "sample_fever.jsonl"
    sample_lines = [
        {"id": 1, "claim": "Rome is the capital of Italy.", "label": "SUPPORTS", "evidence": [["anno_1", 1, "Rome", 0]]},
        {"id": 2, "claim": "Paris is located in Germany.", "label": "REFUTES", "evidence": ["Paris is in France."]},
        {"id": 3, "claim": "Aliens built pyramids.", "label": "NOT ENOUGH INFO", "evidence": []},
    ]
    with open(fever_file, "w", encoding="utf-8") as f:
        for item in sample_lines:
            f.write(json.dumps(item) + "\n")

    loaded = load_fever_dataset(str(fever_file))
    assert len(loaded) == 3
    assert loaded[0]["ground_truth"] == "SUPPORTED"
    assert loaded[1]["ground_truth"] == "REFUTED"
    assert loaded[2]["ground_truth"] == "NOT_ENOUGH_INFORMATION"

    # Evidence is properly structured
    assert len(loaded[0]["evidence"]) >= 1
    assert loaded[0]["evidence"][0]["source_domain"] == "wikipedia.org"


def test_stratified_sample_deterministic_and_balanced():
    dataset = [
        {"id": f"S{i}", "ground_truth": "SUPPORTED"} for i in range(10)
    ] + [
        {"id": f"R{i}", "ground_truth": "REFUTED"} for i in range(10)
    ] + [
        {"id": f"N{i}", "ground_truth": "NOT_ENOUGH_INFORMATION"} for i in range(10)
    ]

    sampled_1 = stratified_sample(dataset, n_samples=6, seed=42)
    sampled_2 = stratified_sample(dataset, n_samples=6, seed=42)

    assert len(sampled_1) == 6
    # Deterministic with fixed seed
    assert [x["id"] for x in sampled_1] == [x["id"] for x in sampled_2]

    # Balanced across 3 classes
    counts = {}
    for item in sampled_1:
        gt = item["ground_truth"]
        counts[gt] = counts.get(gt, 0) + 1
    assert counts.get("SUPPORTED") == 2
    assert counts.get("REFUTED") == 2
    assert counts.get("NOT_ENOUGH_INFORMATION") == 2


def test_split_dev_test_disjoint_and_stratified():
    dataset = [
        {"id": f"S{i}", "ground_truth": "SUPPORTED"} for i in range(20)
    ] + [
        {"id": f"R{i}", "ground_truth": "REFUTED"} for i in range(20)
    ] + [
        {"id": f"N{i}", "ground_truth": "NOT_ENOUGH_INFORMATION"} for i in range(20)
    ]

    dev_set, test_set = split_dev_test(dataset, dev_ratio=0.5, seed=42, stratify=True)

    assert len(dev_set) == 30
    assert len(test_set) == 30

    # Strictly disjoint
    dev_ids = {x["id"] for x in dev_set}
    test_ids = {x["id"] for x in test_set}
    assert dev_ids.isdisjoint(test_ids)

    # Class balance preserved in both splits
    dev_counts = {}
    for x in dev_set:
        dev_counts[x["ground_truth"]] = dev_counts.get(x["ground_truth"], 0) + 1
    assert dev_counts["SUPPORTED"] == 10
    assert dev_counts["REFUTED"] == 10
    assert dev_counts["NOT_ENOUGH_INFORMATION"] == 10
