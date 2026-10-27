"""Tests for the explicit-node causal temporal heterogeneous graph model."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from owerf_temporal_heterograph import (
    NODE_TYPES, RELATION_NAMES, TemporalGraphSchema, TemporalHeteroGraphEncoder,
    causal_snapshot_indices,
)
from owerf_v02_benchmark import generate_synthetic_claim_graph


def _example(seed: int = 7):
    data = generate_synthetic_claim_graph(
        n_samples=800, seed=seed, hidden_mechanisms=("collusion",), tier=2
    )
    schema = TemporalGraphSchema.from_generated_claims(data)
    return data, schema


def test_schema_has_explicit_typed_nodes_and_relations():
    data, schema = _example()
    report = schema.schema_report()
    assert set(report["node_types"]) == set(NODE_TYPES)
    assert len(report["typed_relations"]) == len(RELATION_NAMES) == 6
    assert report["node_types"]["Claim"] == 800
    assert report["node_types"]["Admission"] == 800
    assert "index < start" in report["temporal_policy"]


def test_strict_past_snapshot_indices():
    history, query = causal_snapshot_indices(120, 128)
    assert history.tolist() == list(range(120))
    assert query.tolist() == list(range(120, 128))
    with pytest.raises(ValueError):
        causal_snapshot_indices(128, 120)


def test_temporal_heterograph_forward_shapes_and_gradients():
    data, schema = _example()
    model = TemporalHeteroGraphEncoder(n_features=data["x"].shape[1], n_classes=4)
    x = torch.as_tensor(data["x"], dtype=torch.float32)
    exo = torch.as_tensor(data["exo"], dtype=torch.float32)
    days = torch.as_tensor(schema.days, dtype=torch.float32)
    entity_ids = tuple(torch.as_tensor(item, dtype=torch.long) for item in schema.entity_ids)
    history = torch.arange(0, 120)
    targets = torch.arange(120, 128)
    emb, logits, mu, log_sigma = model(
        x, exo, entity_ids, schema.node_counts, days, history, targets
    )
    assert emb.shape == (8, 48)
    assert logits.shape == (8, 4)
    assert mu.shape == (8, 3) and log_sigma.shape == (8, 3)
    (logits.square().mean() + mu.square().mean()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and
               p.grad.abs().sum() > 0 for p in model.parameters())


def test_future_claim_features_cannot_change_earlier_query_outputs():
    data, schema = _example(seed=11)
    torch.manual_seed(13)
    model = TemporalHeteroGraphEncoder(n_features=data["x"].shape[1], n_classes=4)
    model.eval()
    x = torch.as_tensor(data["x"], dtype=torch.float32)
    exo = torch.as_tensor(data["exo"], dtype=torch.float32)
    days = torch.as_tensor(schema.days, dtype=torch.float32)
    entity_ids = tuple(torch.as_tensor(item, dtype=torch.long) for item in schema.entity_ids)
    history = torch.arange(0, 120)
    targets = torch.arange(120, 128)
    with torch.no_grad():
        original = model(x, exo, entity_ids, schema.node_counts, days, history, targets)[1]
        changed_x = x.clone()
        changed_x[128:] = changed_x[128:] + 1000.0
        changed = model(changed_x, exo, entity_ids, schema.node_counts, days, history, targets)[1]
    assert torch.allclose(original, changed, atol=1e-6)


def test_temporal_leakage_guard_rejects_history_containing_query():
    data, schema = _example(seed=17)
    model = TemporalHeteroGraphEncoder(n_features=data["x"].shape[1], n_classes=4)
    x = torch.as_tensor(data["x"], dtype=torch.float32)
    exo = torch.as_tensor(data["exo"], dtype=torch.float32)
    days = torch.as_tensor(schema.days, dtype=torch.float32)
    entity_ids = tuple(torch.as_tensor(item, dtype=torch.long) for item in schema.entity_ids)
    with pytest.raises(ValueError, match="Temporal leakage"):
        model(x, exo, entity_ids, schema.node_counts, days,
              torch.arange(0, 121), torch.arange(120, 128))
