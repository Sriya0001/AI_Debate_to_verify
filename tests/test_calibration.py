import os
import json
import pytest
import numpy as np

from eval.calibration import (
    Calibrator,
    apply_temperature_scaling,
    fit_temperature,
    fit_abstain_threshold,
    calibrate_dev_and_test,
    compute_nll,
)


def test_temperature_scaling_preserves_probability_simplex():
    probs = np.array([0.7, 0.2, 0.1])
    scaled = apply_temperature_scaling(probs, temperature=1.5)
    assert len(scaled) == 3
    assert np.all(scaled >= 0.0)
    assert abs(np.sum(scaled) - 1.0) < 1e-5

    # High temperature -> more uniform
    high_t = apply_temperature_scaling(probs, temperature=5.0)
    assert high_t[0] < probs[0]
    # Low temperature -> sharper
    low_t = apply_temperature_scaling(probs, temperature=0.5)
    assert low_t[0] > probs[0]


def test_fit_temperature_minimizes_nll():
    # Construct overconfident predictions (need T > 1 to soften)
    probs = np.array([
        [0.95, 0.03, 0.02],
        [0.90, 0.05, 0.05],
        [0.10, 0.85, 0.05],
        [0.20, 0.20, 0.60],
        [0.80, 0.10, 0.10],
    ])
    labels = np.array([0, 0, 1, 0, 1])  # 2 errors out of 5, so 0.95 confidence is overconfident

    nll_t1 = compute_nll(probs, labels, temperature=1.0)
    best_t = fit_temperature(probs, labels)
    nll_best = compute_nll(probs, labels, temperature=best_t)

    assert nll_best <= nll_t1 + 1e-5
    assert best_t > 0.05


def test_fit_abstain_threshold_meets_target_accuracy():
    confidences = np.array([0.95, 0.90, 0.85, 0.75, 0.60, 0.50])
    predictions = np.array([0, 0, 1, 1, 2, 0])
    labels      = np.array([0, 0, 1, 0, 2, 1])  # items 3 and 5 are wrong (conf 0.75, 0.50)

    # If target_accuracy = 0.80:
    # At c >= 0.51, items with conf in {0.95, 0.90, 0.85, 0.75, 0.60} are kept (4 correct out of 5 = 80.0%)
    tau = fit_abstain_threshold(confidences, predictions, labels, target_accuracy=0.80)
    assert tau > 0.50  # Must exclude the 0.50 error
    mask = confidences >= tau
    acc = np.mean(predictions[mask] == labels[mask])
    assert acc >= 0.80

    # If target_accuracy = 1.0 (100%), must filter out the 0.75 error as well (tau > 0.75)
    tau_100 = fit_abstain_threshold(confidences, predictions, labels, target_accuracy=1.0)
    assert tau_100 > 0.75
    mask_100 = confidences >= tau_100
    assert np.mean(predictions[mask_100] == labels[mask_100]) == 1.0


def test_calibrator_save_and_load(tmp_path):
    save_file = str(tmp_path / "test_calibration.json")
    cal = Calibrator(temperature=1.42, abstain_threshold=0.78, target_accuracy=0.92)
    cal.save_to_json(save_file)

    loaded = Calibrator.load_from_json(save_file)
    assert abs(loaded.temperature - 1.42) < 1e-4
    assert abs(loaded.abstain_threshold - 0.78) < 1e-4
    assert abs(loaded.target_accuracy - 0.92) < 1e-4


def test_dev_test_separation_and_output_exposure(tmp_path):
    # DEV split
    dev_records = [
        {"claim_id": "D1", "ground_truth": "SUPPORTED", "final_distribution": [0.90, 0.05, 0.05]},
        {"claim_id": "D2", "ground_truth": "REFUTED", "final_distribution": [0.10, 0.85, 0.05]},
        {"claim_id": "D3", "ground_truth": "SUPPORTED", "final_distribution": [0.60, 0.30, 0.10]},
        {"claim_id": "D4", "ground_truth": "NOT_ENOUGH_INFORMATION", "final_distribution": [0.20, 0.20, 0.60]},
        {"claim_id": "D_FAIL", "ground_truth": "SUPPORTED", "final_distribution": None, "failed": True},
    ]

    # TEST split (never seen during fitting)
    test_records = [
        {"claim_id": "T1", "ground_truth": "SUPPORTED", "final_distribution": [0.95, 0.03, 0.02]},
        {"claim_id": "T2", "ground_truth": "REFUTED", "final_distribution": [0.15, 0.80, 0.05]},
        {"claim_id": "T3", "ground_truth": "SUPPORTED", "final_distribution": [0.45, 0.40, 0.15]},
        {"claim_id": "T_FAIL", "ground_truth": "REFUTED", "final_distribution": None, "failed": True},
    ]

    save_path = str(tmp_path / "calib_params.json")
    calibrator, dev_out, test_out = calibrate_dev_and_test(
        dev_records=dev_records,
        test_records=test_records,
        target_accuracy=0.85,
        save_path=save_path,
    )

    assert os.path.exists(save_path)
    assert calibrator.is_fitted
    assert calibrator.dev_samples_count == 4  # D_FAIL excluded

    # Test output exposure
    for item in test_out:
        if item.get("failed"):
            assert item["abstain"] is True
            assert item["raw_confidence"] is None
            assert item["calibrated_confidence"] is None
        else:
            assert "raw_confidence" in item
            assert "calibrated_confidence" in item
            assert "abstain" in item
            assert isinstance(item["raw_confidence"], float)
            assert isinstance(item["calibrated_confidence"], float)
            assert isinstance(item["abstain"], bool)
            assert "calibrated_distribution" in item
            assert len(item["calibrated_distribution"]) == 3
