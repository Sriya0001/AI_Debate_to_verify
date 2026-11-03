from typing import List, Dict

BENCHMARK_CLAIMS: List[Dict[str, str]] = [
    # ── Category 1: Established True Claims (Ground Truth: SUPPORTED) ────────
    {
        "id": "B1",
        "claim": "The Eiffel Tower is 330 meters tall including antennas and was completed in 1889.",
        "ground_truth": "SUPPORTED",
        "domain": "Architecture & History",
    },
    {
        "id": "B2",
        "claim": "Water boils at 100 degrees Celsius at standard sea-level atmospheric pressure.",
        "ground_truth": "SUPPORTED",
        "domain": "Physics",
    },
    {
        "id": "B3",
        "claim": "The Apollo 11 lunar module landed on the Moon on July 20, 1969.",
        "ground_truth": "SUPPORTED",
        "domain": "Space Exploration",
    },
    {
        "id": "B4",
        "claim": "Alexander Fleming discovered penicillin in 1928.",
        "ground_truth": "SUPPORTED",
        "domain": "Medicine",
    },

    # ── Category 2: Disproven / False Claims (Ground Truth: REFUTED) ──────────
    {
        "id": "B5",
        "claim": "The Moon is primarily made of green cheese and was colonized in 1969.",
        "ground_truth": "REFUTED",
        "domain": "Astronomy",
    },
    {
        "id": "B6",
        "claim": "Humans only use 10 percent of their brain capacity.",
        "ground_truth": "REFUTED",
        "domain": "Neuroscience",
    },
    {
        "id": "B7",
        "claim": "The Great Wall of China is the only human-made structure visible from the Moon with the naked eye.",
        "ground_truth": "REFUTED",
        "domain": "Geography & Space",
    },
    {
        "id": "B8",
        "claim": "Albert Einstein failed mathematics in elementary school.",
        "ground_truth": "REFUTED",
        "domain": "Biography",
    },

    # ── Category 3: Insufficient / Ambiguous Claims (Ground Truth: NOT_ENOUGH_INFORMATION) ─
    {
        "id": "B9",
        "claim": "Project Titan discovered sentient extraterrestrial signals in deep space in March 2026.",
        "ground_truth": "NOT_ENOUGH_INFORMATION",
        "domain": "Astrophysics Rumor",
    },
    {
        "id": "B10",
        "claim": "A secret tunnel directly connects the Pentagon to the Kremlin.",
        "ground_truth": "NOT_ENOUGH_INFORMATION",
        "domain": "Geopolitical Conspiracy",
    },
    {
        "id": "B11",
        "claim": "Quantum computers definitively achieved sentient consciousness in January 2026.",
        "ground_truth": "NOT_ENOUGH_INFORMATION",
        "domain": "Computing Speculation",
    },
    {
        "id": "B12",
        "claim": "Gustave Eiffel secretly built a secondary tower identical to the Eiffel Tower in the Amazon rainforest in 1891.",
        "ground_truth": "NOT_ENOUGH_INFORMATION",
        "domain": "Historical Legend",
    },
]


def load_benchmark_dataset(file_path: str = None) -> List[Dict[str, str]]:
    """
    Loads benchmark claims from an external JSON/CSV file, or returns default 12-claim suite.
    External file format (JSON):
    [{"id": "...", "claim": "...", "ground_truth": "SUPPORTED"|"REFUTED"|"NOT_ENOUGH_INFORMATION", "domain": "..."}]
    """
    if not file_path:
        return BENCHMARK_CLAIMS

    import json
    import csv
    import os

    if not os.path.exists(file_path):
        print(f"Dataset file {file_path} not found. Defaulting to internal benchmark claims.")
        return BENCHMARK_CLAIMS

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    elif file_path.endswith(".csv"):
        items = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                items.append({
                    "id": row.get("id", f"C{idx+1}"),
                    "claim": row.get("claim", row.get("text", "")),
                    "ground_truth": row.get("ground_truth", row.get("label", "NOT_ENOUGH_INFORMATION")).upper(),
                    "domain": row.get("domain", "General"),
                })
        return items

    return BENCHMARK_CLAIMS

