#!/usr/bin/env python3
"""OW-ERF manuscript v0.2: executable causal-relational synthetic pilot.

Stages implemented in this pilot:
1. Timestamped claim/entity graph generation with four fraud mechanisms and benign clinical outliers.
2. Causal relation-context construction (a claim can only aggregate earlier events).
3. Relation-specific attention encoder and episodic pseudo-novel energy-margin training.
4. Energy + Mahalanobis CDF fusion and split-conformal rejection.
5. Legitimate-only clinical expectation and structural reference; explained-rarity filter.
6. Suspicious pool, HDBSCAN grouping when available (DBSCAN fallback), cluster-card summary.
7. Audit metrics and Random Forest / Isolation Forest comparisons.

Scope guard: this is a parametric synthetic pilot. It does not yet invoke Synthea,
use an authoritative transcribed HBP 2.2 package-rate catalogue, or implement a
full PyTorch-Geometric HGT/R-GCN over arbitrary typed nodes. See the data card.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

MECHANISMS = ("upcoding", "unbundling", "phantom_billing", "collusion")
RELATIONS = ("provider", "hospital", "doctor", "beneficiary", "package", "referral_target")
NODE_TYPES = ("Claim", "Beneficiary", "Hospital", "Doctor", "Admission", "Package", "Diagnosis")
HBP_SOURCE = "https://nha.gov.in/img/resources/HBP-2.2-manual.pdf"
SYNTHEA_SOURCE = "https://github.com/synthetichealth/synthea"


def generate_synthetic_claim_graph(
    n_samples: int = 2400,
    seed: int = 42,
    tier: int = 2,
    hidden_mechanisms: Sequence[str] = ("collusion",),
    fraud_rate: float = 0.08,
    benign_outlier_rate: float = 0.015,
) -> dict[str, Any]:
    """Generate timestamped claim events and a hidden ground-truth table separately."""
    if n_samples < 800:
        raise ValueError("n_samples must be at least 800 for temporal train/selection/calibration/test windows.")
    if tier not in (1, 2, 3):
        raise ValueError("tier must be 1, 2 or 3.")
    hidden = tuple(hidden_mechanisms)
    if not hidden or any(x not in MECHANISMS for x in hidden):
        raise ValueError(f"hidden_mechanisms must be a non-empty subset of {MECHANISMS}.")
    if not 0.02 <= fraud_rate <= 0.30:
        raise ValueError("fraud_rate must be between 0.02 and 0.30.")
    if not 0 <= benign_outlier_rate <= 0.10:
        raise ValueError("benign_outlier_rate must be between 0 and 0.10.")

    rng = np.random.default_rng(seed)
    order_time = np.arange(n_samples, dtype=np.int64)
    day = np.linspace(0.0, 1095.0, n_samples, dtype=np.float32)
    deployment_start = int(n_samples * 0.80)
    known_mechanisms = tuple(m for m in MECHANISMS if m not in hidden)

    # The withheld mechanism(s) are generated only in the deployment window;
    # no labelled example of a held-out class exists in train/selection/calibration.
    labels = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        if i < deployment_start:
            choices = ("benign",) + known_mechanisms
            probs = np.asarray([1.0 - fraud_rate] + [fraud_rate / max(1, len(known_mechanisms))] * len(known_mechanisms))
        else:
            choices = ("benign",) + MECHANISMS
            probs = np.asarray([1.0 - fraud_rate] + [fraud_rate / len(MECHANISMS)] * len(MECHANISMS))
        selected = rng.choice(choices, p=probs / probs.sum())
        labels[i] = 0 if selected == "benign" else MECHANISMS.index(selected) + 1

    severity = np.clip(rng.beta(2.0, 4.0, n_samples), 0.02, 0.99)
    comorbidity = np.clip(rng.beta(2.0, 5.0, n_samples), 0.01, 0.99)
    has_organ_support = (rng.random(n_samples) < (0.08 + 0.62 * severity)).astype(float)
    documented_procedures = rng.poisson(1.0 + 4.0 * severity).clip(0, 12).astype(float)
    los_days = np.clip(1 + rng.poisson(2 + 10 * severity + 2 * comorbidity), 1, 90).astype(float)
    billed_tier = np.digitize(severity + rng.normal(0, 0.12, n_samples), bins=[0.3, 0.57, 0.8]).astype(float)
    preauth_ratio = np.clip(rng.normal(0.82, 0.13, n_samples), 0.0, 1.0)
    documentation = np.clip(rng.beta(8, 2, n_samples), 0.0, 1.0)
    has_encounter = np.ones(n_samples, dtype=float)
    procedure_lines = np.clip(documented_procedures + rng.poisson(0.5, n_samples), 0, 18).astype(float)
    duplicate_line_share = np.clip(rng.beta(1.2, 12.0, n_samples), 0.0, 1.0)
    capacity_utilisation = np.clip(0.30 + 0.8 * severity + rng.normal(0, 0.15, n_samples), 0.02, 1.7)
    base_amount = (6500 + 12500 * severity + 5200 * comorbidity + 700 * los_days) * np.exp(rng.normal(0, 0.16, n_samples))
    claimed_amount = np.clip(base_amount, 800, 600000)
    package_rate = np.clip(8000 + 14000 * severity + rng.normal(0, 2500, n_samples), 1500, 80000)

    n_providers = max(20, min(100, n_samples // 25))
    n_hospitals = max(12, int(n_providers * 0.7))
    n_doctors = max(40, int(n_providers * 2.0))
    n_beneficiaries = max(300, int(n_samples * 0.45))
    n_packages = 50

    provider = rng.integers(0, n_providers, n_samples)
    hospital = provider % n_hospitals
    doctor = rng.integers(0, n_doctors, n_samples)
    beneficiary = rng.integers(0, n_beneficiaries, n_samples)
    package = np.clip(np.digitize(severity, bins=np.linspace(0.1, 0.9, n_packages - 1)), 0, n_packages - 1)
    referral_target = rng.integers(0, n_providers, n_samples)

    intensities = {1: 1.0, 2: 0.65, 3: 0.35}
    intensity = intensities[tier]
    is_upcoding = labels == 1
    is_unbundling = labels == 2
    is_phantom = labels == 3
    is_collusion = labels == 4

    # A: package/bed tier and amount are raised beyond clinical expectation.
    claimed_amount[is_upcoding] *= np.exp(rng.normal(0.65 * intensity, 0.12, int(is_upcoding.sum())))
    billed_tier[is_upcoding] = np.minimum(4, billed_tier[is_upcoding] + (1 if tier != 3 else 0.5))
    # B: overlapping / duplicate procedure lines split from an all-inclusive package.
    procedure_lines[is_unbundling] += rng.integers(2, 5, int(is_unbundling.sum()))
    duplicate_line_share[is_unbundling] = np.clip(
        duplicate_line_share[is_unbundling] + rng.normal(0.52 * intensity, 0.08, int(is_unbundling.sum())), 0, 1
    )
    claimed_amount[is_unbundling] *= np.exp(rng.normal(0.22 * intensity, 0.08, int(is_unbundling.sum())))
    # C: billed claim with weak/no encounter provenance.
    has_encounter[is_phantom] = (rng.random(int(is_phantom.sum())) > (0.75 if tier == 1 else 0.45 if tier == 2 else 0.20)).astype(float)
    documentation[is_phantom] = np.clip(
        documentation[is_phantom] - rng.normal(0.55 * intensity, 0.08, int(is_phantom.sum())), 0, 1
    )
    los_days[is_phantom] = np.clip(los_days[is_phantom] * (0.25 if tier == 1 else 0.55 if tier == 2 else 0.8), 1, 90)
    claimed_amount[is_phantom] *= np.exp(rng.normal(0.75 * intensity, 0.12, int(is_phantom.sum())))
    capacity_utilisation[is_phantom] = np.clip(capacity_utilisation[is_phantom] + 0.35 * intensity, 0, 2)
    # D: claims on directed provider-referral cycles; partner IDs form small rings.
    ring_size = int(rng.integers(3, 6))
    ring_starts = np.arange(0, n_providers - ring_size, ring_size)
    ring_map = {int(i + j): int(i + (j + 1) % ring_size) for i in ring_starts for j in range(ring_size)}
    for i in np.flatnonzero(is_collusion):
        if int(provider[i]) not in ring_map:
            provider[i] = int(i % max(1, n_providers - ring_size))
            provider[i] = int(provider[i] // ring_size * ring_size)
        referral_target[i] = ring_map.get(int(provider[i]), int((provider[i] + 1) % n_providers))

    # Benign but clinically extreme claims: severe, well-documented and not fraud-labelled.
    benign_outlier = (labels == 0) & (rng.random(n_samples) < benign_outlier_rate)
    benign_cohort_id = np.zeros(n_samples, dtype=np.int8)
    if benign_outlier.any():
        cohort_choices = rng.choice([1, 2, 3], int(benign_outlier.sum()), p=[0.4, 0.3, 0.3])
        benign_cohort_id[benign_outlier] = cohort_choices
        severity[benign_outlier] = rng.uniform(0.88, 0.99, int(benign_outlier.sum()))
        comorbidity[benign_outlier] = rng.uniform(0.65, 0.95, int(benign_outlier.sum()))
        has_organ_support[benign_outlier] = 1.0
        documented_procedures[benign_outlier] = rng.integers(4, 11, int(benign_outlier.sum()))
        los_days[benign_outlier] = rng.integers(8, 31, int(benign_outlier.sum()))
        documentation[benign_outlier] = rng.uniform(0.88, 1.0, int(benign_outlier.sum()))
        # Recompute an explainable high amount after severity/LOS changes.
        claimed_amount[benign_outlier] = 6500 + 12500 * severity[benign_outlier] + 5200 * comorbidity[benign_outlier] + 700 * los_days[benign_outlier]
        claimed_amount[benign_outlier] *= np.exp(rng.normal(0, 0.08, int(benign_outlier.sum())))

    # Observable claim features. Hidden labels, mechanism names, ring IDs and cohort IDs are excluded.
    feature_names = [
        "severity", "comorbidity", "log_claimed_amount", "log_package_rate",
        "amount_to_package_ratio", "billed_bed_tier", "preauth_ratio",
        "log_length_of_stay", "procedure_lines", "duplicate_line_share",
        "documentation_completeness", "organ_support", "has_encounter",
        "capacity_utilisation",
    ]
    x = np.column_stack([
        severity, comorbidity, np.log1p(claimed_amount), np.log1p(package_rate),
        claimed_amount / np.maximum(package_rate, 1.0), billed_tier, preauth_ratio,
        np.log1p(los_days), procedure_lines, duplicate_line_share, documentation,
        has_organ_support, has_encounter, capacity_utilisation,
    ]).astype(np.float32)
    exo = np.column_stack([
        severity, comorbidity, documented_procedures / 12.0, has_organ_support,
    ]).astype(np.float32)
    target = np.column_stack([
        np.log1p(claimed_amount), billed_tier, np.log1p(los_days),
    ]).astype(np.float32)

    # Causal relational context: each relation reads running aggregates from earlier claims only.
    entity_keys = [provider, hospital, doctor, beneficiary, package, referral_target]
    contexts = np.zeros((n_samples, len(RELATIONS), 7), dtype=np.float32)
    past_observation = np.column_stack([
        severity, np.log1p(claimed_amount) / 14.0, np.log1p(los_days) / 5.0,
        procedure_lines / 18.0, documentation, preauth_ratio,
    ]).astype(np.float32)
    states: list[dict[int, tuple[np.ndarray, int]]] = [dict() for _ in RELATIONS]
    for i in order_time:
        for r, key_arr in enumerate(entity_keys):
            key = int(key_arr[i])
            previous = states[r].get(key)
            if previous is not None:
                sums, count = previous
                contexts[i, r, :6] = sums / count
                contexts[i, r, 6] = np.log1p(count)
        # Update only after every relation for this claim has read the past.
        for r, key_arr in enumerate(entity_keys):
            key = int(key_arr[i])
            previous = states[r].get(key)
            if previous is None:
                states[r][key] = (past_observation[i].copy(), 1)
            else:
                sums, count = previous
                states[r][key] = (sums + past_observation[i], count + 1)

    # Explicit edge list, with all entity relation timestamps recorded. Label table is separate.
    edge_rows = []
    for i in range(n_samples):
        claim_id = f"CLM{i:07d}"
        edge_rows.extend([
            (claim_id, "Claim", claim_id, RELATIONS[j], str(int(entity_keys[j][i])), float(day[i]))
            for j in range(len(RELATIONS))
        ])
    edge_records = [
        {
            "claim_id": row[0], "src_type": row[1], "src_id": row[2],
            "relation_type": row[3], "dst_id": row[4], "timestamp_day": row[5],
        }
        for row in edge_rows
    ]
    label_records = [
        {
            "claim_id": f"CLM{i:07d}",
            "is_fraud": int(labels[i] != 0),
            "mechanism": "benign" if labels[i] == 0 else MECHANISMS[labels[i] - 1],
            "tier": int(tier) if labels[i] != 0 else 0,
            "novelty_status": "held_out" if labels[i] != 0 and MECHANISMS[labels[i] - 1] in hidden else "known_or_benign",
            "benign_cohort_id": int(benign_cohort_id[i]),
            "scheme_instance_id": int((labels[i] - 1) * 100000 + provider[i]) if labels[i] != 0 else -1,
            "ring_id": int(provider[i] // max(3, ring_size)) if is_collusion[i] else -1,
            "injection_seed": int(seed),
        }
        for i in range(n_samples)
    ]
    return {
        "x": x, "exo": exo, "target": target, "context": contexts,
        "labels": labels, "benign_outlier": benign_outlier, "benign_cohort_id": benign_cohort_id,
        "day": day, "provider": provider, "hospital": hospital, "doctor": doctor,
        "beneficiary": beneficiary, "package": package, "referral_target": referral_target,
        "feature_names": feature_names, "edge_records": edge_records, "label_records": label_records,
        "deployment_start_index": deployment_start, "n_nodes_by_type": {
            "Claim": n_samples, "Beneficiary": n_beneficiaries, "Hospital": n_hospitals,
            "Doctor": n_doctors, "Admission": n_samples, "Package": n_packages, "Diagnosis": n_packages,
        },
    }


class CausalRelationalEncoder(nn.Module):
    """Relation-specific message aggregation over prior-window entity summaries."""
    def __init__(self, n_features: int, context_dim: int = 7, n_relations: int = 6,
                 exo_dim: int = 4, n_targets: int = 3, n_classes: int = 4, hidden_dim: int = 48):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(n_features, hidden_dim), nn.ReLU())
        self.relation_layers = nn.ModuleList([nn.Linear(context_dim, hidden_dim) for _ in range(n_relations)])
        self.relation_type = nn.Parameter(torch.randn(n_relations, hidden_dim) * 0.02)
        self.attention = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.fuse = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.expectation = nn.Sequential(nn.Linear(exo_dim, 32), nn.ReLU(), nn.Linear(32, 2 * n_targets))
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor, context: torch.Tensor, exo: torch.Tensor):
        query = self.query(x)
        relation_tokens = torch.stack([
            F.relu(self.relation_layers[r](context[:, r, :])) + self.relation_type[r]
            for r in range(context.shape[1])
        ], dim=1)
        query_expanded = query.unsqueeze(1).expand_as(relation_tokens)
        scores = self.attention(torch.cat([query_expanded, relation_tokens], dim=-1)).squeeze(-1)
        valid = context[:, :, -1] > 0
        # If no past neighbours exist, use a zero message rather than a uniform fake neighbour.
        all_missing = ~valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[all_missing, 0] = True
        scores = scores.masked_fill(~safe_valid, -1e4)
        weights = torch.softmax(scores, dim=1)
        aggregate = (weights.unsqueeze(-1) * relation_tokens).sum(dim=1)
        aggregate[all_missing] = 0.0
        embedding = self.fuse(torch.cat([query, aggregate], dim=1))
        logits = self.classifier(embedding)
        mu, log_sigma = self.expectation(exo).chunk(2, dim=1)
        return embedding, logits, mu, log_sigma.clamp(-3.0, 2.0)


def _energy(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


def _episodic_loss(logits: torch.Tensor, y: torch.Tensor, fraud_class_ids: Sequence[int],
                   m_in: float = -8.0, m_out: float = -2.0, lam: float = 0.05) -> torch.Tensor:
    k_star = int(fraud_class_ids[torch.randint(len(fraud_class_ids), (1,)).item()])
    masked = logits.clone()
    masked[:, k_star] = -1e4
    known = y != k_star
    pseudo_novel = y == k_star
    ce = F.cross_entropy(masked[known], y[known]) if known.any() else logits.sum() * 0.0
    energy = _energy(masked)
    l_in = F.relu(energy[known] - m_in).pow(2).mean() if known.any() else energy.sum() * 0.0
    l_out = F.relu(m_out - energy[pseudo_novel]).pow(2).mean() if pseudo_novel.any() else energy.sum() * 0.0
    return ce + lam * (l_in + l_out)


def _gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (log_sigma + (target - mu).pow(2) / (2 * torch.exp(2 * log_sigma))).mean()


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_ref = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(sorted_ref, values, side="right") / (len(sorted_ref) + 1.0)


def _mahalanobis(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    query = np.asarray(query, dtype=float)
    center = reference.mean(axis=0)
    covariance = np.atleast_2d(np.cov(reference, rowvar=False))
    covariance += np.eye(covariance.shape[0]) * 0.1
    precision = np.linalg.pinv(covariance)
    delta = query - center
    return np.maximum(0.0, np.einsum("bi,ij,bj->b", delta, precision, delta))


def _metrics_for_novel(y_true: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    return {
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, score)) if np.any(y_true == 1) else None,
    }


def _forward(model, x, context, exo, device, batch_size=256):
    model.eval()
    outputs = {"embedding": [], "logits": [], "mu": [], "log_sigma": []}
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xx = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            cc = torch.as_tensor(context[start:start + batch_size], dtype=torch.float32, device=device)
            ee = torch.as_tensor(exo[start:start + batch_size], dtype=torch.float32, device=device)
            e, logits, mu, log_sigma = model(xx, cc, ee)
            for key, val in (("embedding", e), ("logits", logits), ("mu", mu), ("log_sigma", log_sigma)):
                outputs[key].append(val.cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def _cluster_escalations(embeddings: np.ndarray, clinical: np.ndarray, structural: np.ndarray) -> tuple[np.ndarray, str]:
    if len(embeddings) < 5:
        return np.full(len(embeddings), -1, dtype=int), "insufficient_points"
    matrix = np.column_stack([embeddings, clinical, structural])
    matrix = StandardScaler().fit_transform(matrix)
    try:
        import hdbscan  # type: ignore
        labels = hdbscan.HDBSCAN(min_cluster_size=min(8, max(5, len(matrix) // 20)), min_samples=3).fit_predict(matrix)
        return labels.astype(int), "HDBSCAN"
    except ImportError:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=1.6, min_samples=5).fit_predict(matrix)
        return labels.astype(int), "DBSCAN_fallback_HDBSCAN_not_installed"


def run_fold(
    n_samples: int = 2400,
    epochs: int = 3,
    batch_size: int = 128,
    hidden_mechanisms: Sequence[str] = ("collusion",),
    tier: int = 2,
    seed: int = 42,
    alpha: float = 0.05,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run one chronological leave-one-out / leave-two-out OW-ERF pilot fold."""
    hidden = tuple(hidden_mechanisms)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = generate_synthetic_claim_graph(n_samples=n_samples, seed=seed, tier=tier, hidden_mechanisms=hidden)
    n = n_samples
    train_end, val_end, cal_end = int(0.60*n), int(0.70*n), int(0.80*n)
    train_ids = np.arange(0, train_end)
    val_ids = np.arange(train_end, val_end)
    cal_ids = np.arange(val_end, cal_end)
    test_ids = np.arange(cal_end, n)
    labels = data["labels"]
    hidden_ids = np.asarray([MECHANISMS.index(m) + 1 for m in hidden], dtype=int)
    assert not np.isin(labels[np.concatenate([train_ids, val_ids, cal_ids])], hidden_ids).any()

    known_original = [0] + [i for i in range(1, len(MECHANISMS) + 1) if i not in hidden_ids]
    label_map = {original: mapped for mapped, original in enumerate(known_original)}
    mapped = np.full(n, -1, dtype=int)
    for original, mapped_value in label_map.items():
        mapped[labels == original] = mapped_value
    train_y = mapped[train_ids]
    val_mask = mapped[val_ids] >= 0
    cal_mask = mapped[cal_ids] >= 0
    # Only known classes are in selection/calibration by construction.
    val_ids = val_ids[val_mask]
    cal_ids = cal_ids[cal_mask]
    val_y = mapped[val_ids]
    cal_y = mapped[cal_ids]
    if len(np.unique(train_y)) < min(3, len(known_original)):
        raise RuntimeError("Training window does not contain enough known classes.")

    # Fit scalers from the training window only. Clinical expectations are normalized
    # with legitimate training claims only and fitted with a benign-only NLL loss.
    x_scaler = StandardScaler().fit(data["x"][train_ids])
    x_scaled = x_scaler.transform(data["x"]).astype(np.float32)
    exo_train_benign = train_ids[labels[train_ids] == 0]
    exo_scaler = StandardScaler().fit(data["exo"][exo_train_benign])
    exo_scaled = exo_scaler.transform(data["exo"]).astype(np.float32)
    target_mean = data["target"][exo_train_benign].mean(axis=0)
    target_std = data["target"][exo_train_benign].std(axis=0)
    target_std = np.where(target_std < 1e-4, 1.0, target_std)
    target_norm = ((data["target"] - target_mean) / target_std).astype(np.float32)

    class_count = len(known_original)
    model = CausalRelationalEncoder(
        n_features=x_scaled.shape[1], context_dim=data["context"].shape[-1],
        n_relations=data["context"].shape[1], exo_dim=exo_scaled.shape[1],
        n_targets=target_norm.shape[1], n_classes=class_count, hidden_dim=48
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 17)
    fraud_class_ids = list(range(1, class_count))
    best_state = None
    best_val_loss = float("inf")
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_ids)
        losses = []
        for start in range(0, len(shuffled), batch_size):
            batch_ids = shuffled[start:start+batch_size]
            if len(batch_ids) < 8:
                continue
            xx = torch.as_tensor(x_scaled[batch_ids], dtype=torch.float32, device=device)
            cc = torch.as_tensor(data["context"][batch_ids], dtype=torch.float32, device=device)
            ee = torch.as_tensor(exo_scaled[batch_ids], dtype=torch.float32, device=device)
            yy = torch.as_tensor(mapped[batch_ids], dtype=torch.long, device=device)
            tt = torch.as_tensor(target_norm[batch_ids], dtype=torch.float32, device=device)
            optimiser.zero_grad(set_to_none=True)
            _, logits, mu, log_sigma = model(xx, cc, ee)
            class_loss = _episodic_loss(logits, yy, fraud_class_ids)
            benign = yy == 0
            nll = _gaussian_nll(mu[benign], log_sigma[benign], tt[benign]) if benign.any() else mu.sum()*0
            loss = class_loss + 0.25*nll
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; aborting rather than saving invalid results.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise RuntimeError("No training batches ran.")
        epoch_losses.append(float(np.mean(losses)))
        val_out = _forward(model, x_scaled[val_ids], data["context"][val_ids], exo_scaled[val_ids], device)
        val_ce = float(F.cross_entropy(
            torch.as_tensor(val_out["logits"], dtype=torch.float32),
            torch.as_tensor(val_y, dtype=torch.long)
        ).item())
        if val_ce < best_val_loss:
            best_val_loss = val_ce
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    train_out = _forward(model, x_scaled[train_ids], data["context"][train_ids], exo_scaled[train_ids], device)
    val_out = _forward(model, x_scaled[val_ids], data["context"][val_ids], exo_scaled[val_ids], device)
    cal_out = _forward(model, x_scaled[cal_ids], data["context"][cal_ids], exo_scaled[cal_ids], device)
    test_out = _forward(model, x_scaled[test_ids], data["context"][test_ids], exo_scaled[test_ids], device)

    train_y_known = mapped[train_ids]
    class_means = []
    for k in range(class_count):
        class_means.append(train_out["embedding"][train_y_known == k].mean(axis=0))
    train_embed = train_out["embedding"]
    covariance = np.atleast_2d(np.cov(train_embed, rowvar=False))
    covariance += np.eye(covariance.shape[0]) * 0.10
    precision = np.linalg.pinv(covariance)
    def class_mahalanobis(embeddings: np.ndarray) -> np.ndarray:
        scores = []
        for center in class_means:
            delta = embeddings - center
            scores.append(np.maximum(0.0, np.einsum("bi,ij,bj->b", delta, precision, delta)))
        return np.min(np.stack(scores), axis=0)

    train_energy = -np.logaddexp.reduce(train_out["logits"], axis=1)
    val_energy = -np.logaddexp.reduce(val_out["logits"], axis=1)
    cal_energy = -np.logaddexp.reduce(cal_out["logits"], axis=1)
    test_energy = -np.logaddexp.reduce(test_out["logits"], axis=1)
    val_md = class_mahalanobis(val_out["embedding"])
    cal_md = class_mahalanobis(cal_out["embedding"])
    test_md = class_mahalanobis(test_out["embedding"])
    energy_cdf = _ecdf(val_energy, val_energy)
    md_cdf = _ecdf(val_md, val_md)
    val_score = 0.5*energy_cdf + 0.5*md_cdf
    cal_score = 0.5*_ecdf(val_energy, cal_energy) + 0.5*_ecdf(val_md, cal_md)
    test_score = 0.5*_ecdf(val_energy, test_energy) + 0.5*_ecdf(val_md, test_md)
    p_values = (1.0 + (cal_score[:, None] >= test_score[None, :]).sum(axis=0)) / (len(cal_score) + 1.0)
    rejected = p_values <= alpha

    # Clinical and structural thresholds use only benign calibration claims.
    cal_benign_mask = labels[cal_ids] == 0
    cal_benign_ids = cal_ids[cal_benign_mask]
    if len(cal_benign_ids) < 10:
        raise RuntimeError("Need at least 10 benign calibration claims for the explanation thresholds.")
    target_cal = target_norm[cal_ids]
    target_test = target_norm[test_ids]
    sigma_cal = np.exp(np.clip(cal_out["log_sigma"], -3, 2))
    sigma_test = np.exp(np.clip(test_out["log_sigma"], -3, 2))
    clinical_cal = np.maximum((target_cal-cal_out["mu"])/sigma_cal, 0).max(axis=1)
    clinical_test = np.maximum((target_test-test_out["mu"])/sigma_test, 0).max(axis=1)

    structural = data["context"].reshape(n, -1)
    benign_train_structural = structural[exo_train_benign]
    benign_cal_structural = structural[cal_benign_ids]
    struct_cal = _mahalanobis(benign_train_structural, structural[cal_ids])
    struct_test = _mahalanobis(benign_train_structural, structural[test_ids])
    tau_a = max(float(np.quantile(clinical_cal[cal_benign_mask], 0.95)), 1e-6)
    tau_s = max(float(np.quantile(struct_cal[cal_benign_mask], 0.95)), 1e-6)
    unexplained = np.maximum(clinical_test/tau_a, struct_test/tau_s)
    escalated = rejected & (unexplained > 1.0)
    priority = test_score * np.minimum(unexplained, 2.0)

    test_labels = labels[test_ids]
    hidden_mask = np.isin(test_labels, hidden_ids)
    known_test_mask = ~hidden_mask
    benign_clinical_mask = (test_labels == 0) & (data["benign_cohort_id"][test_ids] > 0)
    if not hidden_mask.any():
        raise RuntimeError("Test split has no claims from the held-out mechanism(s).")
    metric = _metrics_for_novel(hidden_mask.astype(int), test_score)
    test_probs = torch.softmax(torch.as_tensor(test_out["logits"]), dim=1).numpy()
    pred_known = test_probs.argmax(axis=1)
    class_y_test = mapped[test_ids]
    valid_known = class_y_test >= 0
    macro_f1_known = float(f1_score(class_y_test[valid_known], pred_known[valid_known], average="macro")) if valid_known.any() else None
    k = min(100, len(test_ids))
    topk = np.argsort(priority)[::-1][:k]
    precision_at_k = float(hidden_mask[topk].mean()) if k else None
    time_to_detection_weeks = None
    hidden_test_indices = np.flatnonzero(hidden_mask)
    alarm_hidden_indices = np.flatnonzero(escalated & hidden_mask)
    if len(alarm_hidden_indices):
        first_hidden_day = float(data["day"][test_ids[hidden_test_indices[0]]])
        first_alert_day = float(data["day"][test_ids[alarm_hidden_indices[0]]])
        time_to_detection_weeks = max(0.0, (first_alert_day - first_hidden_day)/7.0)

    cluster_labels, cluster_backend = _cluster_escalations(
        test_out["embedding"][escalated], clinical_test[escalated], struct_test[escalated]
    )
    if escalated.sum() >= 2 and len(np.unique(cluster_labels)) > 1:
        group_truth = np.where(hidden_mask[escalated], test_labels[escalated], 0)
        # Treat the clusterer's noise as one cluster for reported grouping scores.
        group_pred = cluster_labels.copy()
        group_pred[group_pred < 0] = int(cluster_labels.max() + 1)
        nmi = float(normalized_mutual_info_score(group_truth, group_pred))
        ari = float(adjusted_rand_score(group_truth, group_pred))
    else:
        nmi, ari = None, None

    # Baselines: RF classifies only known classes; IF is fitted only on legitimate training claims.
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    rf.fit(x_scaled[train_ids], train_y_known)
    rf_novelty = 1.0 - rf.predict_proba(x_scaled[test_ids]).max(axis=1)
    if_scores_model = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=1)
    if_scores_model.fit(x_scaled[exo_train_benign])
    if_novelty = -if_scores_model.score_samples(x_scaled[test_ids])

    label_name = np.asarray(["benign"] + list(MECHANISMS), dtype=object)[test_labels]
    test_predictions = []
    for local_i, global_i in enumerate(test_ids):
        test_predictions.append({
            "claim_id": f"CLM{global_i:07d}",
            "timestamp_day": float(data["day"][global_i]),
            "true_mechanism_for_evaluation_only": str(label_name[local_i]),
            "benign_cohort_id_for_evaluation_only": int(data["benign_cohort_id"][global_i]),
            "novelty_score": float(test_score[local_i]),
            "conformal_p_value": float(p_values[local_i]),
            "rejected_at_alpha": bool(rejected[local_i]),
            "clinical_inconsistency": float(clinical_test[local_i]),
            "structural_deviation": float(struct_test[local_i]),
            "unexplainedness_ratio": float(unexplained[local_i]),
            "escalated_to_suspicious_pool": bool(escalated[local_i]),
            "audit_priority": float(priority[local_i]),
            "predicted_known_class": int(pred_known[local_i]),
            "cluster_label": int(cluster_labels[np.flatnonzero(escalated)[list(np.flatnonzero(escalated)).index(local_i)]]) if escalated[local_i] and local_i in np.flatnonzero(escalated) else -1,
        })

    report: dict[str, Any] = {
        "experiment": "OW-ERF v0.2 causal-relational synthetic pilot",
        "manuscript_status": "pre-implementation manuscript translated to pilot code; all measurements here are new pilot outputs",
        "dataset": "parametric synthetic claims with timestamped entity relationships; not Synthea output and not an official AB-PMJAY HBP rate transcription",
        "sources_for_future_grounding": {"Synthea": SYNTHEA_SOURCE, "AB_PMJAY_HBP_2_2": HBP_SOURCE},
        "scope_note": "Causal relation-specific attention over past entity summaries is implemented; full arbitrary-node HGT/R-GCN via PyG and the curated official HBP package-rate table remain future work.",
        "seed": seed, "n_samples": n_samples, "fraud_rate_design_parameter": fraud_rate,
        "camouflage_tier": tier, "hidden_mechanisms": list(hidden),
        "node_types_in_schema": list(NODE_TYPES), "relation_types_in_graph": list(RELATIONS),
        "n_nodes_by_type": data["n_nodes_by_type"], "edge_count": len(data["edge_records"]),
        "train_claims": len(train_ids), "selection_claims": len(val_ids),
        "calibration_claims": len(cal_ids), "deployment_claims": len(test_ids),
        "label_leakage_guard": "Held-out mechanism labels are absent from train/selection/calibration windows; hidden truth is used only for evaluation metrics.",
        "training_epochs": epochs, "device": str(device), "training_loss_first_epoch": epoch_losses[0],
        "training_loss_last_epoch": epoch_losses[-1], "best_validation_cross_entropy": best_val_loss,
        "alpha": alpha, "tau_clinical": tau_a, "tau_structural": tau_s,
        "known_class_macro_f1": macro_f1_known,
        "OW_ERF_novelty_metrics": metric,
        "novel_recall_at_alpha": float(escalated[hidden_mask].mean()),
        "novelty_rejection_recall_before_filter": float(rejected[hidden_mask].mean()),
        "clinical_false_alarm_rate_before_filter": float(rejected[benign_clinical_mask].mean()) if benign_clinical_mask.any() else None,
        "clinical_false_alarm_rate_after_filter": float(escalated[benign_clinical_mask].mean()) if benign_clinical_mask.any() else None,
        "known_class_realised_false_rejection": float(rejected[known_test_mask].mean()) if known_test_mask.any() else None,
        "precision_at_k": {"k": k, "value": precision_at_k},
        "time_to_detection_weeks": time_to_detection_weeks,
        "grouping": {
            "backend": cluster_backend, "escalated_claims": int(escalated.sum()),
            "non_noise_clusters": int(len(set(cluster_labels.tolist()) - {-1})),
            "noise_fraction": float(np.mean(cluster_labels < 0)) if len(cluster_labels) else 1.0,
            "NMI": nmi, "ARI": ari,
        },
        "baselines": {
            "random_forest_one_minus_max_probability": _metrics_for_novel(hidden_mask.astype(int), rf_novelty),
            "isolation_forest_anomaly_score": _metrics_for_novel(hidden_mask.astype(int), if_novelty),
        },
        "artifacts": {},
        "limitations": [
            "Generated data and entity relationships are controlled parametric proxies; not a real claims dataset.",
            "This first runnable graph-style pilot does not invoke Synthea or transcribe the official HBP 2.2 package/rate annex.",
            "Relation-specific attention consumes causal summaries of past entity events; it is not a full arbitrary-node PyG HGT/R-GCN.",
            "One seed and one camouflage tier per run; no claims of statistical significance or publication-ready evidence.",
            "Synthetic benchmark labels are visible to the evaluator but not supplied to the model as features.",
        ],
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "owerf_v02_graph_report.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        prediction_path = out / "owerf_v02_test_predictions.json"
        prediction_path.write_text(json.dumps(test_predictions, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Keep hidden truth separate from observable feature tables; generated data are not committed by CI.
        hidden_truth_path = out / "owerf_v02_hidden_ground_truth.json"
        hidden_truth_path.write_text(json.dumps(data["label_records"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report["artifacts"] = {
            "report": str(report_path), "test_predictions": str(prediction_path),
            "hidden_ground_truth_for_evaluation_only": str(hidden_truth_path),
        }
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def run_matrix(
    n_samples: int = 2400,
    epochs: int = 3,
    batch_size: int = 128,
    hidden: str = "all",
    tier: int = 2,
    seed: int = 42,
    output_dir: str = "results/owerf_v02_pilot",
) -> dict[str, Any]:
    if hidden == "all":
        folds = [(m,) for m in MECHANISMS]
    else:
        folds = [tuple(x.strip() for x in hidden.split(",") if x.strip())]
    if any(not fold or any(m not in MECHANISMS for m in fold) for fold in folds):
        raise ValueError(f"Invalid hidden mechanism setting. Choose one of {MECHANISMS}, comma-separated, or 'all'.")
    root = Path(output_dir)
    reports = []
    for i, fold in enumerate(folds):
        reports.append(run_fold(
            n_samples=n_samples, epochs=epochs, batch_size=batch_size,
            hidden_mechanisms=fold, tier=tier, seed=seed + i,
            output_dir=str(root / ("_".join(fold))),
        ))
    result = {
        "experiment": "OW-ERF v0.2 synthetic benchmark matrix",
        "dataset_scope": "parametric synthetic pilot only",
        "n_samples_per_fold": n_samples, "epochs_per_fold": epochs,
        "base_seed": seed, "folds": reports,
        "limitations": [
            "Single seed per fold; no significance claims.",
            "Does not yet ingest Synthea exports or official HBP package rates.",
            "A full PyG HGT/R-GCN and additional declared baselines/ablations remain future work.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "owerf_v02_synthetic_benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", default="all", help="one mechanism, comma-separated leave-two-out pair, or 'all' for leave-one-out matrix")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/owerf_v02_pilot")
    args = parser.parse_args()
    report = run_matrix(
        n_samples=args.n_samples, epochs=args.epochs, batch_size=args.batch_size,
        hidden=args.hidden, tier=args.tier, seed=args.seed, output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
