"""Tests for the runnable synthetic OW-ERF deep prototype."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from owerf_deep_experiment import generate_synthetic_claims, run_experiment


def test_synthetic_generator_provides_all_encoder_inputs():
    data = generate_synthetic_claims(n_samples=1000, seed=11)
    assert data["hist"].shape == (1000, 10, 6)
    assert data["codes"].shape == (1000, 14)
    assert data["ctx"].shape == (1000, 6, 16)
    assert data["ctx_pad_mask"].shape == (1000, 6)
    assert data["exo"].shape == (1000, 3)
    assert data["targets"].shape == (1000, 2)
    assert data["code_profile"].shape == (1000, 4)
    assert set(np.unique(data["labels"])) == {0, 1, 2, 3, 4}
    assert np.isfinite(data["hist"]).all()
    assert np.isfinite(data["ctx"]).all()


def test_one_fold_training_calibration_and_evaluation_runs():
    report = run_experiment(
        n_samples=1200,
        epochs=2,
        batch_size=128,
        hidden="collusion",
        seed=123,
        output_path=None,
    )
    assert report["n_train_known"] > 0
    assert report["n_calibration_known"] > 0
    assert report["n_test"] > 0
    assert report["hidden_mechanism"] == "collusion"
    assert report["classes_seen_in_training"] == ["benign", "upcoding", "unbundling", "phantom_billing"]
    assert 0.0 <= report["multimodal_novelty_roc_auc"] <= 1.0
    assert report["multimodal_unseen_recall_after_explained_rarity_filter"] > 0.0
    assert 0.0 <= report["raw_input_profile_roc_auc"] <= 1.0
    assert 0.0 <= report["isolation_forest_novelty_roc_auc"] <= 1.0
    assert np.isfinite(report["training_loss_first_epoch"])
    assert np.isfinite(report["training_loss_last_epoch"])
    assert "Synthetic proof-of-pipeline only" in report["limitations"][0]
