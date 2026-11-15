import json
import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar

from config import config

logger = logging.getLogger(__name__)

CLASS_NAMES = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def label_to_idx(label: Union[str, int]) -> int:
    """Normalizes ground truth / prediction label to an integer index {0, 1, 2}."""
    if isinstance(label, (int, np.integer)):
        return int(label)
    label_str = str(label).strip().upper()
    if label_str in CLASS_TO_IDX:
        return CLASS_TO_IDX[label_str]
    # Handle minor variants
    if label_str in ("SUPPORTS", "SUPPORT"):
        return CLASS_TO_IDX["SUPPORTED"]
    if label_str in ("REFUTES", "REFUTE"):
        return CLASS_TO_IDX["REFUTED"]
    if label_str in ("NOT ENOUGH INFO", "NOT_ENOUGH_INFO", "NEI", "UNKNOWN"):
        return CLASS_TO_IDX["NOT_ENOUGH_INFORMATION"]
    raise ValueError(f"Unknown class label '{label}'. Expected one of {CLASS_NAMES}.")


def apply_temperature_scaling(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """
    Applies temperature scaling per docs/FCMAD_MATH.md Section 6:
        p^{cal}_k = (p^*_k)^{1/T} / sum_l (p^*_l)^{1/T}
    """
    probs = np.asarray(probabilities, dtype=float)
    T = max(float(temperature), 1e-4)

    # Use log-domain computation for numerical stability
    # log(p^{1/T}) = (1/T) * log(p)
    clipped_probs = np.clip(probs, 1e-15, 1.0)
    scaled_logits = np.log(clipped_probs) / T

    if scaled_logits.ndim == 1:
        max_logit = np.max(scaled_logits)
        exp_logits = np.exp(scaled_logits - max_logit)
        return exp_logits / np.sum(exp_logits)
    else:
        max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def compute_nll(probabilities: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """
    Computes Negative Log-Likelihood (NLL) for a given temperature T:
        NLL(T) = - sum_n ln p^{cal}_n[y_n]
    """
    scaled_probs = apply_temperature_scaling(probabilities, temperature)
    n = len(labels)
    if n == 0:
        return 0.0

    picked_probs = scaled_probs[np.arange(n), labels]
    return float(-np.sum(np.log(np.clip(picked_probs, 1e-15, 1.0))))


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bounds: Tuple[float, float] = (0.05, 10.0),
) -> float:
    """
    Finds optimal temperature T minimizing NLL on the given split:
        T = argmin_T - sum_n ln p^{cal}_n[y_n]
    """
    if len(labels) == 0:
        return 1.0

    def objective(t_val: float) -> float:
        return compute_nll(probabilities, labels, t_val)

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    if res.success:
        return float(round(float(res.x), 4))
    return 1.0


def fit_abstain_threshold(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    target_accuracy: float = 0.90,
) -> float:
    """
    Finds optimal abstain threshold tau on DEV split per docs/FCMAD_MATH.md Section 7:
        tau = min { c in [0, 1] : (sum_{c_n >= c} I(y_hat_n == y_n)) / (sum_{c_n >= c} 1) >= target_accuracy }
    If no threshold meets target accuracy, returns 1.0 (abstain on all).
    """
    if len(confidences) == 0:
        return 1.0

    confs = np.asarray(confidences, dtype=float)
    preds = np.asarray(predictions, dtype=int)
    gts = np.asarray(labels, dtype=int)

    # Candidate thresholds from unique confidences plus boundary grid
    candidates = np.unique(np.clip(confs, 0.0, 1.0))
    candidates = np.sort(np.unique(np.concatenate([candidates, np.linspace(0.0, 1.0, 101)])))

    valid_thresholds: List[float] = []
    for c in candidates:
        mask = confs >= c
        total = np.sum(mask)
        if total == 0:
            continue
        correct = np.sum(preds[mask] == gts[mask])
        acc = correct / total
        if acc >= target_accuracy:
            valid_thresholds.append(float(c))

    if valid_thresholds:
        return float(round(min(valid_thresholds), 4))

    # If no threshold meets target accuracy, return 1.0
    return 1.0


class Calibrator:
    """
    Manages temperature calibration and abstain threshold estimation.
    Strictly guarantees that calibration is fit ONLY on DEV split and applied to TEST.
    """
    def __init__(
        self,
        temperature: float = 1.0,
        abstain_threshold: float = 0.60,
        target_accuracy: float = 0.90,
    ):
        self.temperature = float(temperature)
        self.abstain_threshold = float(abstain_threshold)
        self.target_accuracy = float(target_accuracy)
        self.is_fitted = False
        self.dev_samples_count = 0

    def fit_on_dev(
        self,
        dev_records: List[Dict[str, Any]],
        target_accuracy: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fits temperature T and abstain threshold tau strictly on DEV split data.
        Failed items (record['failed'] == True) are strictly excluded from fitting.
        """
        tgt_acc = target_accuracy if target_accuracy is not None else self.target_accuracy
        self.target_accuracy = float(tgt_acc)

        # Exclude failed items
        valid_records = [
            r for r in dev_records
            if not r.get("failed", False) and r.get("ground_truth") is not None and (r.get("probabilities") is not None or r.get("final_distribution") is not None)
        ]

        if not valid_records:
            logger.warning("Calibrator: No valid dev records for calibration; retaining default parameters.")
            return {
                "temperature": self.temperature,
                "abstain_threshold": self.abstain_threshold,
                "target_accuracy": self.target_accuracy,
                "dev_samples": 0,
            }

        probs_list: List[List[float]] = []
        labels_list: List[int] = []

        for r in valid_records:
            dist = r.get("final_distribution") or r.get("probabilities")
            # Normalize to 3 elements summing to 1
            dist_arr = np.asarray(dist, dtype=float)
            if dist_arr.sum() > 0:
                dist_arr = dist_arr / dist_arr.sum()
            else:
                dist_arr = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
            probs_list.append(dist_arr.tolist())
            labels_list.append(label_to_idx(r["ground_truth"]))

        probs = np.array(probs_list, dtype=float)
        labels = np.array(labels_list, dtype=int)

        # 1. Fit Temperature T by minimizing NLL on DEV
        best_T = fit_temperature(probs, labels)
        self.temperature = best_T

        # 2. Compute calibrated probabilities and confidences on DEV
        cal_probs = apply_temperature_scaling(probs, self.temperature)
        cal_confs = np.max(cal_probs, axis=-1)
        preds = np.argmax(cal_probs, axis=-1)

        # 3. Fit Abstain Threshold tau from target accuracy on DEV
        tau = fit_abstain_threshold(cal_confs, preds, labels, target_accuracy=self.target_accuracy)
        self.abstain_threshold = tau

        self.is_fitted = True
        self.dev_samples_count = len(valid_records)

        logger.info(
            "Calibrator fitted on %d DEV samples: T=%.4f, tau=%.4f (target_acc=%.1f%%)",
            self.dev_samples_count, self.temperature, self.abstain_threshold, self.target_accuracy * 100
        )

        return {
            "temperature": self.temperature,
            "abstain_threshold": self.abstain_threshold,
            "target_accuracy": self.target_accuracy,
            "dev_samples": self.dev_samples_count,
        }

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies frozen calibration parameters (T, tau) to a dataset (DEV or TEST).
        Populates raw_confidence, calibrated_confidence, abstain, and final_distribution.
        Failed records are preserved with failed=True without calculating confidence/abstain.
        """
        enriched: List[Dict[str, Any]] = []
        for r in records:
            item = dict(r)
            if item.get("failed", False):
                item["raw_confidence"] = None
                item["calibrated_confidence"] = None
                item["abstain"] = True
                enriched.append(item)
                continue

            dist = item.get("final_distribution") or item.get("probabilities")
            if dist is None:
                item["raw_confidence"] = None
                item["calibrated_confidence"] = None
                item["abstain"] = True
                enriched.append(item)
                continue

            dist_arr = np.asarray(dist, dtype=float)
            if dist_arr.sum() > 0:
                dist_arr = dist_arr / dist_arr.sum()
            else:
                dist_arr = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

            raw_conf = float(round(float(np.max(dist_arr)), 4))

            # Apply temperature scaling
            cal_dist = apply_temperature_scaling(dist_arr, self.temperature)
            cal_conf = float(round(float(np.max(cal_dist)), 4))
            abstain_flag = bool(cal_conf < self.abstain_threshold)

            pred_idx = int(np.argmax(cal_dist))
            pred_label = IDX_TO_CLASS[pred_idx]

            item["raw_confidence"] = raw_conf
            item["calibrated_confidence"] = cal_conf
            item["abstain"] = abstain_flag
            item["calibrated_distribution"] = [float(round(v, 4)) for v in cal_dist]
            item["predicted_label"] = item.get("predicted_label") or pred_label
            enriched.append(item)

        return enriched

    def save_to_json(self, file_path: str):
        """Saves fitted calibration parameters to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "temperature": self.temperature,
            "abstain_threshold": self.abstain_threshold,
            "target_accuracy": self.target_accuracy,
            "is_fitted": self.is_fitted,
            "dev_samples_count": self.dev_samples_count,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved calibration parameters to %s", file_path)

    @classmethod
    def load_from_json(cls, file_path: str) -> "Calibrator":
        """Loads calibration parameters from JSON."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Calibration parameters file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        calibrator = cls(
            temperature=data.get("temperature", 1.0),
            abstain_threshold=data.get("abstain_threshold", 0.60),
            target_accuracy=data.get("target_accuracy", 0.90),
        )
        calibrator.is_fitted = data.get("is_fitted", True)
        calibrator.dev_samples_count = data.get("dev_samples_count", 0)
        logger.info("Loaded calibration parameters from %s: T=%.4f, tau=%.4f", file_path, calibrator.temperature, calibrator.abstain_threshold)
        return calibrator


def calibrate_dev_and_test(
    dev_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    target_accuracy: float = 0.90,
    save_path: Optional[str] = None,
) -> Tuple[Calibrator, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fits calibration parameters strictly on dev_records, optionally saves to JSON,
    and applies them separately to both dev_records and test_records.
    Guarantees that test_records are NEVER seen during parameter fitting.
    """
    calibrator = Calibrator(target_accuracy=target_accuracy)
    calibrator.fit_on_dev(dev_records, target_accuracy=target_accuracy)

    if save_path:
        calibrator.save_to_json(save_path)

    dev_enriched = calibrator.apply(dev_records)
    test_enriched = calibrator.apply(test_records)

    return calibrator, dev_enriched, test_enriched
