"""Tests for the executable OW-ERF v0.2 causal-relational pilot."""
from __future__ import annotations
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from owerf_v02_benchmark import (
    CausalRelationalEncoder,
    RELATIONS,
    generate_synthetic_claim_graph,
    _explained_rarity_gate,
    run_fold,
)


def test_graph_schema_and_heldout_labels_absent_from_predeployment_windows():
    data = generate_synthetic_claim_graph(
        n_samples=1000, seed=13, hidden_mechanisms=("collusion",), tier=2
    )
    assert data["x"].shape[0] == 1000
    assert data["context"].shape == (1000, len(RELATIONS), 7)
    assert data["edge_records"] and data["label_records"]
    cut = data["deployment_start_index"]
    assert not np.any(data["labels"][:cut] == 4)
    assert data["labels"][cut:].max() == 4
    # The first event has no historical neighbours for any relation.
    assert np.all(data["context"][0, :, -1] == 0)
    assert data["n_nodes_by_type"]["Claim"] == 1000


def test_relation_encoder_forward_shapes_and_backward():
    model = CausalRelationalEncoder(n_features=14, n_classes=4)
    x = torch.randn(12, 14)
    context = torch.randn(12, len(RELATIONS), 7)
    context[:, :, -1] = 1
    exo = torch.randn(12, 4)
    emb, logits, mu, log_sigma = model(x, context, exo)
    assert emb.shape == (12, 48)
    assert logits.shape == (12, 4)
    assert mu.shape == (12, 3) and log_sigma.shape == (12, 3)
    (logits.square().mean() + mu.square().mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_end_to_end_temporal_open_world_fold_runs(tmp_path):
    report = run_fold(
        n_samples=1200, epochs=1, batch_size=128,
        hidden_mechanisms=("collusion",), seed=21,
        output_dir=str(tmp_path),
    )
    assert report["hidden_mechanisms"] == ["collusion"]
    assert report["train_claims"] + report["selection_claims"] + report["calibration_claims"] + report["deployment_claims"] == 1200
    assert 0 <= report["OW_ERF_novelty_metrics"]["roc_auc"] <= 1
    assert 0 <= report["novel_recall_at_alpha"] <= 1
    assert (tmp_path / "owerf_v02_graph_report.json").exists()
    assert (tmp_path / "owerf_v02_test_predictions.json").exists()


def test_context_aggregates_only_preceding_claim_events_and_edges_match_timestamps():
    data = generate_synthetic_claim_graph(
        n_samples=1000, seed=31, hidden_mechanisms=("collusion",), tier=2
    )
    relation_keys = [
        data["provider"], data["hospital"], data["doctor"],
        data["beneficiary"], data["package"], data["referral_target"],
    ]
    prior_counts = [dict() for _ in RELATIONS]
    for i in range(1000):
        for relation_index, keys in enumerate(relation_keys):
            key = int(keys[i])
            expected_prior_count = prior_counts[relation_index].get(key, 0)
            assert np.isclose(
                data["context"][i, relation_index, -1],
                np.log1p(expected_prior_count),
            ), f"relation={RELATIONS[relation_index]}, claim={i}"
        # Context was read before any relation state for the current event is updated.
        for relation_index, keys in enumerate(relation_keys):
            key = int(keys[i])
            prior_counts[relation_index][key] = prior_counts[relation_index].get(key, 0) + 1

    for edge in data["edge_records"]:
        claim_index = int(edge["claim_id"][3:])
        assert np.isclose(edge["timestamp_day"], float(data["day"][claim_index]))
        assert edge["relation_type"] in RELATIONS


def test_explained_rarity_gate_requires_both_rejection_and_unexplainedness():
    rejected = np.array([True, True, False, True, True])
    clinical = np.array([1.0, 1.0001, 50.0, 0.2, 0.1])
    structural = np.array([0.3, 0.4, 50.0, 2.0, 0.9])
    unexplainedness, escalated = _explained_rarity_gate(
        rejected, clinical, structural, tau_clinical=1.0, tau_structural=1.0
    )
    assert np.allclose(unexplainedness, [1.0, 1.0001, 50.0, 2.0, 0.9])
    assert escalated.tolist() == [False, True, False, True, False]


def test_explained_rarity_gate_rejects_invalid_thresholds_and_shapes():
    with pytest.raises(ValueError, match="positive"):
        _explained_rarity_gate(np.array([True]), np.array([2.0]), np.array([1.0]), 0.0, 1.0)
    with pytest.raises(ValueError, match="identical lengths"):
        _explained_rarity_gate(np.array([True, False]), np.array([2.0]), np.array([1.0]), 1.0, 1.0)
