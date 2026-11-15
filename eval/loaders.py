import json
import csv
import os
import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from eval.dataset import BENCHMARK_CLAIMS

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "SUPPORT": "SUPPORTED",
    "SUPPORTED": "SUPPORTED",
    "REFUTES": "REFUTED",
    "REFUTE": "REFUTED",
    "REFUTED": "REFUTED",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFORMATION",
    "NOT ENOUGH INFORMATION": "NOT_ENOUGH_INFORMATION",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFORMATION",
    "NOT_ENOUGH_INFORMATION": "NOT_ENOUGH_INFORMATION",
    "NEI": "NOT_ENOUGH_INFORMATION",
}


def load_smoke_test(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads the original 12-claim benchmark suite as a fast smoke test."""
    claims = [dict(c) for c in BENCHMARK_CLAIMS]
    if limit is not None:
        return claims[:limit]
    return claims


def stratified_sample(
    dataset: List[Dict[str, Any]],
    n_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Performs fixed-seed stratified sampling across ground_truth classes."""
    if n_samples >= len(dataset):
        return list(dataset)

    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for item in dataset:
        gt = item.get("ground_truth", "NOT_ENOUGH_INFORMATION")
        by_label.setdefault(gt, []).append(item)

    num_classes = len(by_label)
    if num_classes == 0:
        return []

    per_class = n_samples // num_classes
    remainder = n_samples % num_classes

    sampled: List[Dict[str, Any]] = []
    sorted_labels = sorted(by_label.keys())

    for idx, lbl in enumerate(sorted_labels):
        items = list(by_label[lbl])
        rng.shuffle(items)
        take = per_class + (1 if idx < remainder else 0)
        sampled.extend(items[:take])

    rng.shuffle(sampled)
    return sampled


def load_fever_dataset(
    file_path: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
    stratify: bool = True,
) -> List[Dict[str, Any]]:
    """
    Loads FEVER dataset from a JSONL file with gold evidence.
    Strictly adheres to policy: if file does not exist, raises FileNotFoundError;
    NEVER fabricates synthetic FEVER data.

    Expected JSONL fields:
        id: int or str
        claim: str
        label: "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"
        evidence: list of evidence sentence annotations
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"FEVER dataset file not found at: '{file_path}'. "
            "Please provide the dataset file (e.g., fever_dev.jsonl) in the data/ directory."
        )

    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            raw_label = str(data.get("label", "")).strip().upper()
            if raw_label not in LABEL_MAP:
                continue

            # Format evidence snippets if provided
            evidence_snippets: List[Dict[str, Any]] = []
            raw_ev = data.get("evidence", [])
            if isinstance(raw_ev, list):
                # Flatten any nested lists down to individual items/tuples
                def extract_evidence_items(obj):
                    if isinstance(obj, list):
                        if len(obj) >= 4 and isinstance(obj[2], str) and isinstance(obj[3], (int, str)):
                            # Matched [anno_id, ev_id, wiki_title, sent_id]
                            title, sent_id = obj[2], obj[3]
                            if title:
                                evidence_snippets.append({
                                    "evidence_id": f"E{len(evidence_snippets)+1}",
                                    "text": f"{title} (Sentence {sent_id})",
                                    "snippet": f"Document: {title}, sentence index: {sent_id}",
                                    "source_domain": "wikipedia.org",
                                })
                        else:
                            for item in obj:
                                extract_evidence_items(item)
                    elif isinstance(obj, str) and obj.strip():
                        evidence_snippets.append({
                            "evidence_id": f"E{len(evidence_snippets)+1}",
                            "text": obj.strip(),
                            "snippet": obj.strip(),
                            "source_domain": "wikipedia.org",
                        })
                    elif isinstance(obj, dict):
                        evidence_snippets.append({
                            "evidence_id": obj.get("evidence_id", f"E{len(evidence_snippets)+1}"),
                            "text": obj.get("text", obj.get("snippet", "")),
                            "snippet": obj.get("snippet", obj.get("text", "")),
                            "source_domain": obj.get("source_domain", "wikipedia.org"),
                        })

                extract_evidence_items(raw_ev)

            records.append({
                "id": str(data.get("id", f"FEVER_{line_num}")),
                "claim": data.get("claim", "").strip(),
                "ground_truth": LABEL_MAP[raw_label],
                "evidence": evidence_snippets,
                "domain": "FEVER Wikipedia",
            })

    if max_samples and max_samples < len(records):
        if stratify:
            records = stratified_sample(records, max_samples, seed=seed)
        else:
            rng = random.Random(seed)
            records = rng.sample(records, max_samples)

    logger.info("Loaded %d FEVER claims from %s", len(records), file_path)
    return records


def load_xfact_dataset(
    file_path: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
    stratify: bool = True,
) -> List[Dict[str, Any]]:
    """
    Optional loader for X-Fact multilingual fact-checking dataset.
    Raises FileNotFoundError if file is missing; never fabricates data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"X-Fact dataset file not found at: '{file_path}'.")

    records: List[Dict[str, Any]] = []
    # TSV or JSONL
    if file_path.endswith(".jsonl") or file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                records.append({
                    "id": str(d.get("id", "")),
                    "claim": d.get("claim", ""),
                    "ground_truth": LABEL_MAP.get(str(d.get("label", "")).upper(), "NOT_ENOUGH_INFORMATION"),
                    "evidence": d.get("evidence", []),
                    "domain": "X-Fact",
                })
    else:
        # TSV format
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                records.append({
                    "id": row.get("id", f"XFACT_{idx+1}"),
                    "claim": row.get("claim", ""),
                    "ground_truth": LABEL_MAP.get(str(row.get("label", "")).upper(), "NOT_ENOUGH_INFORMATION"),
                    "evidence": [],
                    "domain": "X-Fact",
                })

    if max_samples and max_samples < len(records):
        if stratify:
            records = stratified_sample(records, max_samples, seed=seed)
        else:
            rng = random.Random(seed)
            records = rng.sample(records, max_samples)

    return records


def split_dev_test(
    dataset: List[Dict[str, Any]],
    dev_ratio: float = 0.5,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Explicitly partitions a dataset into strictly disjoint DEV and TEST sets.
    Performs fixed-seed stratified split across ground_truth classes.
    """
    if not dataset:
        return [], []

    rng = random.Random(seed)

    if not stratify:
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        split_idx = int(len(dataset) * dev_ratio)
        dev_indices = set(indices[:split_idx])
        dev_set = [dataset[i] for i in range(len(dataset)) if i in dev_indices]
        test_set = [dataset[i] for i in range(len(dataset)) if i not in dev_indices]
        return dev_set, test_set

    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for item in dataset:
        gt = item.get("ground_truth", "NOT_ENOUGH_INFORMATION")
        by_label.setdefault(gt, []).append(item)

    dev_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []

    for lbl, items in sorted(by_label.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_dev = int(round(len(shuffled) * dev_ratio))
        dev_set.extend(shuffled[:n_dev])
        test_set.extend(shuffled[n_dev:])

    # Shuffle both final splits with reproducible seed
    rng.shuffle(dev_set)
    rng.shuffle(test_set)

    logger.info(
        "Dataset split (seed=%d): %d total -> %d DEV (%.1f%%), %d TEST (%.1f%%)",
        seed, len(dataset), len(dev_set), len(dev_set) / len(dataset) * 100,
        len(test_set), len(test_set) / len(dataset) * 100
    )

    return dev_set, test_set
