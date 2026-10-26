"""End-to-end OW-ERF deep prototype on generated synthetic claims.

This is a runnable research prototype, not a real-insurance-data benchmark.
It generates temporal histories, procedure-code sequences and relational-context
tokens; trains the LSTM/CNN/Transformer encoder; calibrates novelty using a held
out known-class calibration set; applies the explained-rarity gate; and compares
novelty ranking with an Isolation Forest baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from owerf_deep_encoders import OWERFEncoder, energy, episodic_loss, gaussian_nll

MECHANISMS = ("upcoding", "unbundling", "phantom_billing", "collusion")
LABEL_TO_NAME = {0: "benign", **{i + 1: name for i, name in enumerate(MECHANISMS)}}
HISTORY_LEN = 10
CODE_LEN = 14
CONTEXT_TOKENS = 6
HISTORY_DIM = 6
CONTEXT_DIM = 16
CODE_VOCAB = 500


def generate_synthetic_claims(
    n_samples: int = 4000,
    seed: int = 42,
    fraud_rate: float = 0.12,
    rare_severe_rate: float = 0.05,
) -> dict[str, np.ndarray]:
    """Create reproducible multimodal claims with four named synthetic mechanisms."""
    if n_samples < 800:
        raise ValueError("n_samples must be at least 800 for train/calibration/test splits.")
    if not 0 < fraud_rate < 0.5:
        raise ValueError("fraud_rate must be between 0 and 0.5.")
    if not 0 <= rare_severe_rate < 0.5:
        raise ValueError("rare_severe_rate must be between 0 and 0.5.")

    rng = np.random.default_rng(seed)
    labels = np.zeros(n_samples, dtype=np.int64)
    n_fraud = max(4, int(round(n_samples * fraud_rate)))
    fraud_indices = rng.choice(n_samples, size=n_fraud, replace=False)
    fraud_labels = np.resize(np.arange(1, len(MECHANISMS) + 1), n_fraud)
    rng.shuffle(fraud_labels)
    labels[fraud_indices] = fraud_labels

    severity = rng.beta(2.0, 4.0, size=n_samples)
    comorbidity = rng.beta(2.0, 5.0, size=n_samples)
    age_scaled = rng.uniform(18.0, 90.0, size=n_samples) / 90.0
    rare_severe = (labels == 0) & (rng.random(n_samples) < rare_severe_rate)
    severity[rare_severe] = rng.uniform(0.82, 0.99, size=int(rare_severe.sum()))

    log_amount = 8.2 + 1.15 * severity + 0.35 * comorbidity + rng.normal(0, 0.22, n_samples)
    log_los = 0.15 + 1.1 * severity + 0.35 * comorbidity + rng.normal(0, 0.24, n_samples)

    # Mechanism 1: amount is inflated relative to the exogenous severity profile.
    log_amount[labels == 1] += rng.normal(1.0, 0.18, int((labels == 1).sum()))
    # Mechanism 2: code sequences contain repeated/unusually high-numbered codes.
    # Mechanism 3: a high amount is paired with an implausibly short length of stay.
    phantom = labels == 3
    log_amount[phantom] += rng.normal(0.95, 0.2, int(phantom.sum()))
    log_los[phantom] -= rng.normal(0.95, 0.15, int(phantom.sum()))

    exo = np.column_stack([severity, comorbidity, age_scaled]).astype(np.float32)
    targets = np.column_stack([log_amount, log_los]).astype(np.float32)

    # Recent provider/beneficiary claim history, ordered oldest to newest.
    hist = np.empty((n_samples, HISTORY_LEN, HISTORY_DIM), dtype=np.float32)
    hist[:, :, 0] = 1.15 * severity[:, None] + rng.normal(0, 0.25, (n_samples, HISTORY_LEN))
    hist[:, :, 1] = 1.1 * severity[:, None] + rng.normal(0, 0.25, (n_samples, HISTORY_LEN))
    hist[:, :, 2] = severity[:, None] + rng.normal(0, 0.12, (n_samples, HISTORY_LEN))
    hist[:, :, 3] = rng.beta(2, 4, (n_samples, HISTORY_LEN))
    hist[:, :, 4] = rng.uniform(0, 1, (n_samples, HISTORY_LEN))
    hist[:, :, 5] = 0.5 * comorbidity[:, None] + rng.normal(0, 0.2, (n_samples, HISTORY_LEN))
    collusion = labels == 4
    hist[collusion, :, 5] += 0.65

    # Ordered procedure/line-item codes. Zero is reserved for padding.
    codes = rng.integers(1, 151, size=(n_samples, CODE_LEN), dtype=np.int64)
    for row in np.flatnonzero(labels == 2):
        package_code = int(rng.integers(250, CODE_VOCAB))
        codes[row, ::2] = package_code
        codes[row, 1::2] = rng.integers(250, CODE_VOCAB, size=CODE_LEN // 2)

    # Tokens represent claim, hospital, doctor, package, diagnosis and neighbours.
    # The first token is the claim token and carries the observed current claim
    # values plus exogenous context. This lets the classifier see cost/LOS while
    # the separate expectation head learns whether those values are explainable.
    ctx = rng.normal(0, 0.25, size=(n_samples, CONTEXT_TOKENS, CONTEXT_DIM)).astype(np.float32)
    ctx[:, 0, 3] = severity
    ctx[:, 0, 4] = comorbidity
    ctx[:, 0, 5] = age_scaled
    ctx[:, 0, 6] = (log_amount - 8.2) / 1.5
    ctx[:, 0, 7] = log_los / 2.0
    ctx[:, 0, 8] = (codes > 200).mean(axis=1)
    ctx[collusion, :, 0] += 1.5
    ctx[collusion, :, 1] += 1.2
    ctx[collusion, :, 2] += 0.8
    ctx_pad_mask = np.zeros((n_samples, CONTEXT_TOKENS), dtype=np.bool_)
    pad_counts = rng.integers(0, 3, size=n_samples)
    for row, count in enumerate(pad_counts):
        if count:
            ctx_pad_mask[row, -int(count):] = True

    tabular = np.column_stack([
        exo,
        targets,
        hist.mean(axis=1),
        ctx.mean(axis=1),
        (codes > 200).mean(axis=1),
    ]).astype(np.float32)

    return {
        "hist": hist,
        "codes": codes,
        "ctx": ctx,
        "ctx_pad_mask": ctx_pad_mask,
        "exo": exo,
        "targets": targets,
        "labels": labels,
        "rare_severe": rare_severe,
        "tabular": tabular,
    }


def _stratified_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        ids = rng.permutation(ids)
        n_train = max(1, int(0.65 * len(ids)))
        n_cal = max(1, int(0.15 * len(ids)))
        if n_train + n_cal >= len(ids):
            n_train = max(1, len(ids) - 2)
            n_cal = 1
        train_parts.append(ids[:n_train])
        calibration_parts.append(ids[n_train:n_train + n_cal])
        test_parts.append(ids[n_train + n_cal:])
    return (
        rng.permutation(np.concatenate(train_parts)),
        rng.permutation(np.concatenate(calibration_parts)),
        rng.permutation(np.concatenate(test_parts)),
    )


def _inputs(data: dict[str, np.ndarray], ids: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(data["hist"][ids], dtype=torch.float32, device=device),
        torch.as_tensor(data["codes"][ids], dtype=torch.long, device=device),
        torch.as_tensor(data["ctx"][ids], dtype=torch.float32, device=device),
        torch.as_tensor(data["ctx_pad_mask"][ids], dtype=torch.bool, device=device),
        torch.as_tensor(data["exo"][ids], dtype=torch.float32, device=device),
    )


def _outputs(
    model: OWERFEncoder,
    data: dict[str, np.ndarray],
    ids: np.ndarray,
    device: torch.device,
    targets_norm: np.ndarray,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    model.eval()
    pieces: dict[str, list[np.ndarray]] = {k: [] for k in ("h", "logits", "mu", "log_sigma")}
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            *inputs, _unused = (_inputs(data, batch_ids, device) + (
                torch.as_tensor(targets_norm[batch_ids], dtype=torch.float32, device=device),
            ))
            h, logits, mu, log_sigma = model(*inputs)
            for key, value in (("h", h), ("logits", logits), ("mu", mu), ("log_sigma", log_sigma)):
                pieces[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in pieces.items()}


def _mahalanobis(
    reference_h: np.ndarray,
    reference_y: np.ndarray,
    query_h: np.ndarray,
) -> np.ndarray:
    """Minimum regularized Mahalanobis distance to the known-class centroids."""
    covariance = np.cov(reference_h, rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(covariance.shape[0]) * 0.05
    precision = np.linalg.pinv(covariance)
    distances = []
    for label in np.unique(reference_y):
        center = reference_h[reference_y == label].mean(axis=0)
        delta = query_h - center
        dist = np.einsum("bi,ij,bj->b", delta, precision, delta)
        distances.append(dist)
    return np.min(np.stack(distances, axis=0), axis=0)


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(sorted_reference, values, side="right") / (len(sorted_reference) + 1.0)


def _inconsistency(targets: np.ndarray, mu: np.ndarray, log_sigma: np.ndarray) -> np.ndarray:
    sigma = np.exp(np.clip(log_sigma, -3.0, 2.0))
    standardized = (targets - mu) / sigma
    return np.maximum(standardized, 0.0).max(axis=1)


def run_experiment(
    n_samples: int = 4000,
    epochs: int = 12,
    batch_size: int = 128,
    hidden: str = "collusion",
    seed: int = 42,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Train on three known mechanisms and test novelty against one held-out mechanism."""
    if hidden not in MECHANISMS:
        raise ValueError(f"hidden must be one of {MECHANISMS}; got {hidden!r}")
    if epochs < 1 or batch_size < 8:
        raise ValueError("epochs must be >= 1 and batch_size must be >= 8.")

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = generate_synthetic_claims(n_samples=n_samples, seed=seed)
    train_all, calibration_all, test_ids = _stratified_split(data["labels"], seed + 1)

    hidden_id = MECHANISMS.index(hidden) + 1
    train_ids = train_all[data["labels"][train_all] != hidden_id]
    calibration_ids = calibration_all[data["labels"][calibration_all] != hidden_id]
    known_original = [0] + [i for i in range(1, len(MECHANISMS) + 1) if i != hidden_id]
    label_map = {original: mapped for mapped, original in enumerate(known_original)}
    mapped_y = np.full(n_samples, -1, dtype=np.int64)
    for original, mapped in label_map.items():
        mapped_y[data["labels"] == original] = mapped
    train_y = mapped_y[train_ids]
    calibration_y = mapped_y[calibration_ids]

    if len(np.unique(train_y)) < 4:
        raise RuntimeError("Training split must contain benign claims and all three known mechanisms.")
    if len(np.unique(data["labels"][test_ids])) < 2 or not np.any(data["labels"][test_ids] == hidden_id):
        raise RuntimeError("Test split does not contain sufficient class diversity.")

    # Normalize targets using benign training rows only; they are the expectation model's reference.
    benign_train_ids = train_ids[data["labels"][train_ids] == 0]
    target_mean = data["targets"][benign_train_ids].mean(axis=0)
    target_std = data["targets"][benign_train_ids].std(axis=0)
    target_std = np.where(target_std < 1e-4, 1.0, target_std)
    targets_norm = ((data["targets"] - target_mean) / target_std).astype(np.float32)

    model = OWERFEncoder(n_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 2)
    loss_history: list[float] = []

    for _epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_ids)
        epoch_losses = []
        for start in range(0, len(shuffled), batch_size):
            ids = shuffled[start:start + batch_size]
            if len(ids) < 8:
                continue
            histories, codes, context, pad_mask, exo = _inputs(data, ids, device)
            yb = torch.as_tensor(mapped_y[ids], dtype=torch.long, device=device)
            target = torch.as_tensor(targets_norm[ids], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            _, logits, mu, log_sigma = model(histories, codes, context, pad_mask, exo)
            classification_loss = episodic_loss(
                logits, yb, known_fraud_ids=(1, 2, 3), m_in=-8.0, m_out=-2.0, lam=0.05
            )
            benign_mask = yb == 0
            if benign_mask.any():
                regression_loss = gaussian_nll(mu[benign_mask], log_sigma[benign_mask], target[benign_mask])
            else:
                regression_loss = mu.sum() * 0.0
            loss = classification_loss + 0.25 * regression_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered during prototype training.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses:
            raise RuntimeError("No training batches were processed.")
        loss_history.append(float(np.mean(epoch_losses)))

    train_out = _outputs(model, data, train_ids, device, targets_norm)
    cal_out = _outputs(model, data, calibration_ids, device, targets_norm)
    test_out = _outputs(model, data, test_ids, device, targets_norm)

    train_energy = energy(torch.as_tensor(train_out["logits"])).cpu().numpy()
    cal_energy = energy(torch.as_tensor(cal_out["logits"])).cpu().numpy()
    test_energy = energy(torch.as_tensor(test_out["logits"])).cpu().numpy()
    train_md = _mahalanobis(train_out["h"], train_y, train_out["h"])
    cal_md = _mahalanobis(train_out["h"], train_y, cal_out["h"])
    test_md = _mahalanobis(train_out["h"], train_y, test_out["h"])

    novelty_reference_energy = train_energy
    novelty_reference_md = train_md
    train_score = 0.5 * _ecdf(novelty_reference_energy, train_energy) + 0.5 * _ecdf(novelty_reference_md, train_md)
    calibration_score = 0.5 * _ecdf(novelty_reference_energy, cal_energy) + 0.5 * _ecdf(novelty_reference_md, cal_md)
    test_score = 0.5 * _ecdf(novelty_reference_energy, test_energy) + 0.5 * _ecdf(novelty_reference_md, test_md)

    calibration_inconsistency = _inconsistency(
        targets_norm[calibration_ids], cal_out["mu"], cal_out["log_sigma"]
    )
    calibration_benign = data["labels"][calibration_ids] == 0
    if not calibration_benign.any():
        raise RuntimeError("Calibration split must contain benign claims.")
    tau = float(np.quantile(calibration_inconsistency[calibration_benign], 0.95))
    test_inconsistency = _inconsistency(
        targets_norm[test_ids], test_out["mu"], test_out["log_sigma"]
    )

    alpha = 0.05
    p_values = (1.0 + (calibration_score[:, None] >= test_score[None, :]).sum(axis=0)) / (len(calibration_score) + 1.0)
    novelty_flag = p_values <= alpha
    escalate = novelty_flag & (test_inconsistency > tau)

    test_labels = data["labels"][test_ids]
    is_hidden = test_labels == hidden_id
    is_benign = test_labels == 0
    rare_test = data["rare_severe"][test_ids]
    hidden_recall = float(novelty_flag[is_hidden].mean())
    hidden_recall_after_filter = float(escalate[is_hidden].mean())

    forest = IsolationForest(n_estimators=120, contamination="auto", random_state=seed, n_jobs=1)
    forest.fit(data["tabular"][train_ids])
    train_if_score = -forest.score_samples(data["tabular"][train_ids])
    test_if_score = -forest.score_samples(data["tabular"][test_ids])
    hidden_binary = is_hidden.astype(np.int64)

    report: dict[str, Any] = {
        "prototype": "OW-ERF synthetic multimodal training and evaluation",
        "dataset": "generated synthetic claims only; no real patient/insurance data",
        "hidden_mechanism": hidden,
        "seed": seed,
        "n_samples": int(n_samples),
        "epochs": int(epochs),
        "device": str(device),
        "n_train_known": int(len(train_ids)),
        "n_calibration_known": int(len(calibration_ids)),
        "n_test": int(len(test_ids)),
        "classes_seen_in_training": [LABEL_TO_NAME[x] for x in known_original],
        "training_loss_first_epoch": loss_history[0],
        "training_loss_last_epoch": loss_history[-1],
        "alpha": alpha,
        "explained_rarity_threshold_tau": tau,
        "deep_novelty_roc_auc": float(roc_auc_score(hidden_binary, test_score)),
        "deep_novelty_average_precision": float(average_precision_score(hidden_binary, test_score)),
        "deep_unseen_recall_at_alpha": hidden_recall,
        "deep_unseen_recall_after_explained_rarity_filter": hidden_recall_after_filter,
        "deep_benign_rejection_rate_at_alpha": float(novelty_flag[is_benign].mean()) if is_benign.any() else None,
        "deep_benign_escalation_rate_after_filter": float(escalate[is_benign].mean()) if is_benign.any() else None,
        "deep_rare_severe_benign_escalation_rate": float(escalate[rare_test].mean()) if rare_test.any() else None,
        "isolation_forest_novelty_roc_auc": float(roc_auc_score(hidden_binary, test_if_score)),
        "isolation_forest_average_precision": float(average_precision_score(hidden_binary, test_if_score)),
        "isolation_forest_unseen_recall_at_matched_training_alert_rate": float(
            (test_if_score >= np.quantile(train_if_score, 1.0 - alpha))[is_hidden].mean()
        ),
        "limitations": [
            "Synthetic proof-of-pipeline only; no real insurance claims or patient data.",
            "The temporal histories, procedure codes and relation tokens are generated proxies, not extracted from a real claims dataset.",
            "This prototype does not implement a full temporal heterogeneous graph encoder.",
            "One held-out mechanism is evaluated per run; a publication-grade study still needs repeated seeds, all-mechanism ablations and stronger baselines.",
        ],
    }
    if not all(np.isfinite(value) for value in report.values() if isinstance(value, (float, int))):
        raise RuntimeError("Non-finite metric produced; refusing to write an invalid report.")
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["output_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", choices=MECHANISMS, default="collusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/deep_prototype_results.json")
    args = parser.parse_args()
    result = run_experiment(
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
