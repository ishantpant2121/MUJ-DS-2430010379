"""Tests for separately trainable LSTM/CNN/Transformer classifiers."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
torch = pytest.importorskip("torch")

from scripts.train_evaluate_deep_models import (
    CNNClaimClassifier,
    LSTMClaimClassifier,
    TransformerClaimClassifier,
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
