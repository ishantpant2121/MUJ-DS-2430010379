"""Temporal heterogeneous graph encoder for synthetic open-world claim research.

This is a dependency-light, explicit typed-node implementation in PyTorch. It
uses eight node types, eight typed claim-to-entity relation channels, causal history
messages, relation-specific R-GCN-style transforms, time encodings and
attention-based HGT-style relation fusion. It is a research baseline, not a
validated production fraud detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NODE_TYPES = (
    "Claim", "Beneficiary", "Hospital", "Doctor", "Provider", "Admission", "Package", "Diagnosis"
)
RELATION_SPECS = (
    ("provider", "Provider", "provider"),
    ("hospital", "Hospital", "hospital"),
    ("doctor", "Doctor", "doctor"),
    ("beneficiary", "Beneficiary", "beneficiary"),
    ("admission", "Admission", "admission"),
    ("package", "Package", "package"),
    ("diagnosis", "Diagnosis", "diagnosis"),
    ("referral_target", "Provider", "referral_target"),
)
RELATION_NAMES = tuple(item[0] for item in RELATION_SPECS)


@dataclass(frozen=True)
class TemporalGraphSchema:
    """Typed graph indexes derived only from observable synthetic claim fields."""
    entity_ids: tuple[np.ndarray, ...]
    node_counts: tuple[int, ...]
    days: np.ndarray
    feature_names: tuple[str, ...]

    @classmethod
    def from_generated_claims(cls, data: dict[str, Any]) -> "TemporalGraphSchema":
        n = int(len(data["x"]))
        diagnosis = np.clip(
            np.digitize(np.asarray(data["x"])[:, 0], bins=[0.25, 0.50, 0.75]), 0, 3
        ).astype(np.int64)
        provider_ids = np.asarray(data["provider"], dtype=np.int64)
        referral_ids = np.asarray(data["referral_target"], dtype=np.int64)
        provider_count = int(max(provider_ids.max(initial=0), referral_ids.max(initial=0))) + 1
        ids = (
            provider_ids,
            np.asarray(data["hospital"], dtype=np.int64),
            np.asarray(data["doctor"], dtype=np.int64),
            np.asarray(data["beneficiary"], dtype=np.int64),
            np.arange(n, dtype=np.int64),  # one synthetic admission node per claim
            np.asarray(data["package"], dtype=np.int64),
            diagnosis,
            referral_ids,
        )
        counts = (
            provider_count,
            int(data["n_nodes_by_type"]["Hospital"]),
            int(data["n_nodes_by_type"]["Doctor"]),
            int(data["n_nodes_by_type"]["Beneficiary"]),
            n,
            int(data["n_nodes_by_type"]["Package"]),
            int(diagnosis.max()) + 1,
            provider_count,
        )
        for relation, entity_ids, count in zip(RELATION_NAMES, ids, counts):
            if entity_ids.shape != (n,) or entity_ids.min(initial=0) < 0 or entity_ids.max(initial=0) >= count:
                raise ValueError(f"Invalid node indices for relation {relation}")
        return cls(
            entity_ids=ids,
            node_counts=counts,
            days=np.asarray(data["day"], dtype=np.float32),
            feature_names=tuple(data["feature_names"]),
        )

    def schema_report(self) -> dict[str, Any]:
        return {
            "node_types": {
                "Claim": int(len(self.days)),
                **{spec[1]: int(count) for spec, count in zip(RELATION_SPECS, self.node_counts)},
            },
            "typed_relations": [
                {"source": "Claim", "relation": rel, "target": node_type,
                 "direction": "historical Claim -> entity; entity -> query Claim",
                 "timestamped": True}
                for rel, node_type, _ in RELATION_SPECS
            ],
            "temporal_policy": (
                "For a query block [start, end), entity summaries use only claim events "
                "with index < start. Query claims do not message into entity nodes in the "
                "same block. This conservative snapshot policy prevents within-block "
                "future leakage."
            ),
            "diagnosis_proxy": (
                "Diagnosis nodes are synthetic severity-bin proxies because the source "
                "generator does not provide actual diagnosis codes."
            ),
            "admission_proxy": "One synthetic admission node is assigned per claim.",
        }


class TemporalHeteroGraphEncoder(nn.Module):
    """Causal relation-specific message passing over a typed temporal claim graph.

    Each entity node starts with a learned node-type embedding and a log-history
    degree feature, then aggregates transformed historical claim messages with
    recency weighting. Query claims receive typed entity messages and attend over
    the six relation channels. No claim label is part of an input node feature.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 48,
                 exo_dim: int = 4, n_targets: int = 3, dropout: float = 0.10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.claim_in = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.node_type_embedding = nn.Embedding(len(NODE_TYPES), hidden_dim)
        nn.init.normal_(self.node_type_embedding.weight, std=0.02)
        self.degree_projection = nn.Linear(1, hidden_dim, bias=False)
        self.time_projection = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh())
        self.claim_to_entity = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in RELATION_SPECS
        )
        self.entity_norm = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in RELATION_SPECS)
        self.entity_to_claim = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in RELATION_SPECS
        )
        self.relation_embedding = nn.Parameter(torch.randn(len(RELATION_SPECS), hidden_dim) * 0.02)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.expectation = nn.Sequential(
            nn.Linear(exo_dim, 32), nn.GELU(), nn.Linear(32, 2 * n_targets)
        )
        self.n_targets = n_targets

    def forward(
        self,
        x_all: torch.Tensor,
        exo_all: torch.Tensor,
        entity_ids: tuple[torch.Tensor, ...],
        node_counts: tuple[int, ...],
        days: torch.Tensor,
        history_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(entity_ids) != len(RELATION_SPECS):
            raise ValueError("entity_ids must contain an index vector for each typed relation")
        if len(target_indices) == 0:
            raise ValueError("target_indices must not be empty")
        if len(history_indices) and int(history_indices.max()) >= int(target_indices.min()):
            raise ValueError("Temporal leakage: history indices must precede query indices")

        claim_h = self.claim_in(x_all)
        query_h = claim_h[target_indices]
        history_h = claim_h[history_indices]
        query_day = days[target_indices].min()
        time_span = torch.clamp(days.max() - days.min(), min=1.0)

        relation_messages = []
        for r, (relation, node_type, _) in enumerate(RELATION_SPECS):
            count = int(node_counts[r])
            all_ids = entity_ids[r]
            hist_ids = all_ids[history_indices]
            # Only past claim events contribute messages to entity nodes.
            degree = torch.zeros(count, device=x_all.device, dtype=x_all.dtype)
            mass = torch.zeros_like(degree)
            aggregate = torch.zeros(count, self.hidden_dim, device=x_all.device, dtype=x_all.dtype)
            if len(history_indices):
                age = torch.clamp((query_day - days[history_indices]) / time_span, min=0.0, max=1.0)
                time_features = torch.stack((torch.log1p(age * 100.0), torch.exp(-3.0 * age)), dim=1)
                decay = torch.exp(-2.0 * age)
                message = self.claim_to_entity[r](history_h) + self.time_projection(time_features)
                aggregate.index_add_(0, hist_ids, message * decay.unsqueeze(1))
                mass.index_add_(0, hist_ids, decay)
                degree.index_add_(0, hist_ids, torch.ones_like(decay))
                aggregate = aggregate / mass.clamp_min(1e-6).unsqueeze(1)
            type_index = NODE_TYPES.index(node_type)
            type_h = self.node_type_embedding.weight[type_index].unsqueeze(0).expand(count, -1)
            degree_feature = torch.log1p(degree).unsqueeze(1)
            entity_h = self.entity_norm[r](type_h + self.degree_projection(degree_feature) + aggregate)
            query_entity_h = entity_h[all_ids[target_indices]]
            typed_message = self.entity_to_claim[r](query_entity_h) + self.relation_embedding[r]
            rel_embed = self.relation_embedding[r].unsqueeze(0).expand(len(target_indices), -1)
            score = self.attention(torch.cat(
                (query_h, typed_message, rel_embed), dim=1
            ))
            relation_messages.append((score, typed_message))

        scores = torch.cat([item[0] for item in relation_messages], dim=1)
        weights = torch.softmax(scores, dim=1)
        messages = torch.stack([item[1] for item in relation_messages], dim=1)
        aggregated = (weights.unsqueeze(2) * messages).sum(dim=1)
        embedding = self.fusion(torch.cat((query_h, aggregated), dim=1))
        logits = self.classifier(embedding)
        mu, log_sigma = self.expectation(exo_all[target_indices]).chunk(2, dim=1)
        return embedding, logits, mu, log_sigma.clamp(-3.0, 2.0)


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (log_sigma + (target - mu).square() / (2.0 * torch.exp(2.0 * log_sigma))).mean()


def causal_snapshot_indices(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Return strict-past source and current query indices for [start, end)."""
    if start < 0 or end <= start:
        raise ValueError("Expected 0 <= start < end")
    return np.arange(start, dtype=np.int64), np.arange(start, end, dtype=np.int64)
