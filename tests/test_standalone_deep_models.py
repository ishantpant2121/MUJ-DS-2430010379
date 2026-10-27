"""Tests for separately trainable LSTM/CNN/Transformer classifiers."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
torch = pytest.importorskip("torch")

from owerf_deep_experiment import generate_synthetic_claims

from scripts.train_evaluate_deep_models import (
    CNNClaimClassifier,
    CNNContinuousContextClassifier,
    LSTMClaimClassifier,
    TransformerClaimClassifier,
    _prepare_inputs,
    run_benchmark,
)


def test_three_independent_encoders_have_trainable_gradients():
    lstm = LSTMClaimClassifier(input_dim=6, n_classes=4)
    _, logits = lstm(torch.randn(8, 10, 6))
    logits.mean().backward()
    assert logits.shape == (8, 4)
    assert any(p.grad is not None for p in lstm.parameters())

    cnn = CNNClaimClassifier(vocab_size=500, n_classes=4)
    _, logits = cnn(torch.randint(0, 500, (8, 14)))
    logits.mean().backward()
    assert logits.shape == (8, 4)
    assert any(p.grad is not None for p in cnn.parameters())

    transformer = TransformerClaimClassifier(input_dim=16, n_classes=4)
    tokens = torch.randn(8, 6, 16)
    mask = torch.tensor([[False, False, False, False, False, True]] * 8)
    _, logits = transformer(tokens, mask)
    logits.mean().backward()
    assert logits.shape == (8, 4)
    assert any(p.grad is not None for p in transformer.parameters())


def test_small_leave_one_out_benchmark_trains_all_three_models(tmp_path):
    report = run_benchmark(
        n_samples=1200, epochs=1, batch_size=128, seeds=(17,),
        output_dir=str(tmp_path), save_weights=True,
        hidden_mechanisms=("collusion",),
    )
    assert len(report["runs"]) == 3
    assert {row["model"] for row in report["runs"]} == {"LSTM", "CNN1D", "Transformer"}
    for row in report["runs"]:
        assert 0 <= row["known_class_accuracy"] <= 1
        assert 0 <= row["known_class_macro_f1"] <= 1
        assert 0 <= row["novel_mechanism_auroc"] <= 1
        assert 0 <= row["novel_recall_at_conformal_alpha_005"] <= 1
        assert row["trained_weights_path"] is not None
    assert (tmp_path / "standalone_deep_model_metrics.json").exists()
    assert (tmp_path / "standalone_deep_model_metrics.csv").exists()
    assert len(list((tmp_path / "weights").glob("*.pt"))) == 3


def test_shared_context_mode_gives_all_architectures_identical_nonleaking_inputs():
    data = generate_synthetic_claims(n_samples=1000, seed=29)
    train_ids = np.arange(600)
    prepared = [
        _prepare_inputs(name, data, train_ids, input_mode="shared_context")
        for name in ("LSTM", "CNN1D", "Transformer")
    ]
    expected_x = prepared[0][0]["x"]
    expected_mask = prepared[0][0]["padding_mask"]
    for model_inputs, metadata in prepared:
        assert np.array_equal(model_inputs["x"], expected_x)
        assert np.array_equal(model_inputs["padding_mask"], expected_mask)
        assert metadata["input_mode"] == "shared_context"
        assert metadata["padding_mask_used"] is True
    assert np.all(expected_x[expected_mask] == 0.0)

    valid_training_tokens = data["ctx"][train_ids][~data["ctx_pad_mask"][train_ids].astype(bool)]
    expected_scaler = StandardScaler().fit(valid_training_tokens)
    assert np.allclose(prepared[0][1]["scaler_mean"], expected_scaler.mean_)
    assert np.allclose(prepared[0][1]["scaler_scale"], expected_scaler.scale_)


def test_continuous_cnn_masks_padding_and_backpropagates():
    model = CNNContinuousContextClassifier(input_dim=16, n_classes=4)
    x = torch.randn(8, 6, 16)
    mask = torch.tensor([[False, False, False, False, True, True]] * 8)
    _, logits = model(x, mask)
    assert logits.shape == (8, 4)
    logits.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_shared_context_benchmark_trains_all_architectures(tmp_path):
    report = run_benchmark(
        n_samples=1200, epochs=1, batch_size=128, seeds=(19,),
        output_dir=str(tmp_path), save_weights=True,
        hidden_mechanisms=("collusion",), input_mode="shared_context",
    )
    assert report["input_mode"] == "shared_context"
    assert report["comparison_design"].startswith("controlled")
    assert len(report["runs"]) == 3
    assert all(row["input_mode"] == "shared_context" for row in report["runs"])
    assert all(row["input_modality"] == "standardized_relational_context_tokens_shared_by_all_architectures" for row in report["runs"])
    assert len(list((tmp_path / "weights").glob("shared_context_*.pt"))) == 3
