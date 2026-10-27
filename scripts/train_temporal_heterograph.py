#!/usr/bin/env python3
"""Train/evaluate the causal temporal heterogeneous graph OW-ERF research baseline.

Outputs are synthetic-only. The generator deliberately places held-out mechanism
examples after the 80% deployment boundary. All scalers and model parameters are
fit on the chronological training window; calibration is separate and known-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from owerf_temporal_heterograph import (
    RELATION_NAMES, TemporalGraphSchema, TemporalHeteroGraphEncoder, gaussian_nll
)
from owerf_v02_benchmark import MECHANISMS, generate_synthetic_claim_graph

METRICS = (
    "known_accuracy", "known_macro_f1", "novel_auroc", "novel_ap",
    "novel_recall_at_alpha", "known_false_rejection_at_alpha",
    "rare_severe_benign_false_rejection", "gated_novel_recall", "escalation_rate"
)


def batches(start: int, end: int, size: int) -> Iterable[tuple[int, int]]:
    for left in range(start, end, size):
        yield left, min(end, left + size)


def ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ref, values, side="right") / (len(ref) + 1.0)


def mahalanobis(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    center = reference.mean(axis=0)
    covariance = np.atleast_2d(np.cov(reference, rowvar=False))
    covariance += np.eye(covariance.shape[0]) * 0.2
    precision = np.linalg.pinv(covariance)
    delta = values - center
    return np.maximum(0.0, np.einsum("bi,ij,bj->b", delta, precision, delta))


def tensorize(data: dict[str, Any], schema: TemporalGraphSchema, train_end: int,
              device: torch.device) -> dict[str, Any]:
    x_scaler = StandardScaler().fit(data["x"][:train_end])
    exo_scaler = StandardScaler().fit(data["exo"][:train_end])
    benign_train = (data["labels"][:train_end] == 0)
    if int(benign_train.sum()) < 10:
        raise ValueError("Insufficient benign training claims for expectation calibration.")
    target_scaler = StandardScaler().fit(data["target"][:train_end][benign_train])
    return {
        "x": torch.as_tensor(x_scaler.transform(data["x"]), dtype=torch.float32, device=device),
        "exo": torch.as_tensor(exo_scaler.transform(data["exo"]), dtype=torch.float32, device=device),
        "target": torch.as_tensor(target_scaler.transform(data["target"]), dtype=torch.float32, device=device),
        "entity_ids": tuple(torch.as_tensor(a, dtype=torch.long, device=device) for a in schema.entity_ids),
        "node_counts": schema.node_counts,
        "days": torch.as_tensor(schema.days, dtype=torch.float32, device=device),
        "x_scaler": x_scaler,
        "exo_scaler": exo_scaler,
        "target_scaler": target_scaler,
    }


def run_range(model: TemporalHeteroGraphEncoder, arrays: dict[str, Any],
              start: int, end: int, batch_size: int) -> dict[str, np.ndarray]:
    """Evaluate a chronological range with strictly past-only graph snapshots."""
    model.eval()
    out: dict[str, list[np.ndarray]] = {
        "embedding": [], "logits": [], "mu": [], "log_sigma": []
    }
    with torch.no_grad():
        for left, right in batches(start, end, batch_size):
            history = torch.arange(left, device=arrays["x"].device)
            targets = torch.arange(left, right, device=arrays["x"].device)
            emb, logits, mu, log_sigma = model(
                arrays["x"], arrays["exo"], arrays["entity_ids"], arrays["node_counts"],
                arrays["days"], history, targets
            )
            for key, value in (("embedding", emb), ("logits", logits), ("mu", mu),
                               ("log_sigma", log_sigma)):
                out[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in out.items()}


def train_fold(*, n_samples: int, hidden_mechanism: str, seed: int, epochs: int,
               batch_size: int, patience: int, output_dir: Path, device: torch.device) -> dict[str, Any]:
    data = generate_synthetic_claim_graph(
        n_samples=n_samples, seed=seed, hidden_mechanisms=(hidden_mechanism,), tier=2
    )
    schema = TemporalGraphSchema.from_generated_claims(data)
    train_end = int(n_samples * 0.55)
    selection_end = int(n_samples * 0.65)
    calibration_end = int(n_samples * 0.80)
    hidden_label = MECHANISMS.index(hidden_mechanism) + 1
    known_labels = [0] + [i + 1 for i, name in enumerate(MECHANISMS) if name != hidden_mechanism]
    label_to_class = {label: i for i, label in enumerate(known_labels)}
    class_to_label = np.asarray(known_labels, dtype=np.int64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    if not set(np.unique(labels[:selection_end])).issubset(set(known_labels)):
        raise AssertionError("Held-out mechanism contaminated pre-deployment labels")
    y_mapped = np.full(n_samples, -1, dtype=np.int64)
    for label, class_id in label_to_class.items():
        y_mapped[labels == label] = class_id
    arrays = tensorize(data, schema, train_end, device)
    y_tensor = torch.as_tensor(y_mapped, dtype=torch.long, device=device)
    targets = arrays["target"]
    benign_train_idx = np.flatnonzero((labels[:train_end] == 0))
    if len(benign_train_idx) == 0:
        raise ValueError("No benign training claims")

    model = TemporalHeteroGraphEncoder(
        n_features=arrays["x"].shape[1], n_classes=len(known_labels), hidden_dim=48,
        exo_dim=arrays["exo"].shape[1], n_targets=targets.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_values = []
        for left, right in batches(0, train_end, batch_size):
            h = torch.arange(left, device=device)
            q = torch.arange(left, right, device=device)
            emb, logits, mu, log_sigma = model(
                arrays["x"], arrays["exo"], arrays["entity_ids"], arrays["node_counts"],
                arrays["days"], h, q
            )
            yy = y_tensor[q]
            classification = F.cross_entropy(logits, yy)
            benign_mask = yy == 0
            expectation_loss = (
                gaussian_nll(mu[benign_mask], log_sigma[benign_mask], targets[q][benign_mask])
                if bool(benign_mask.any()) else logits.sum() * 0.0
            )
            loss = classification + 0.25 * expectation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss_values.append(float(loss.detach().cpu()))

        val_parts = run_range(model, arrays, train_end, selection_end, batch_size)
        val_labels = y_mapped[train_end:selection_end]
        val_loss = float(F.cross_entropy(
            torch.as_tensor(val_parts["logits"], dtype=torch.float32),
            torch.as_tensor(val_labels, dtype=torch.long)
        ).item())
        train_loss = float(np.mean(train_loss_values))
        history_log.append({"epoch": epoch, "train_loss": train_loss, "selection_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)

    # All representations are recalculated with the selected checkpoint and a
    # causal graph snapshot for each target block.
    train_out = run_range(model, arrays, 0, train_end, batch_size)
    cal_out = run_range(model, arrays, selection_end, calibration_end, batch_size)
    test_out = run_range(model, arrays, calibration_end, n_samples, batch_size)
    train_energy = -np.logaddexp.reduce(train_out["logits"], axis=1)
    cal_energy = -np.logaddexp.reduce(cal_out["logits"], axis=1)
    test_energy = -np.logaddexp.reduce(test_out["logits"], axis=1)

    train_benign = labels[:train_end] == 0
    train_benign_emb = train_out["embedding"][train_benign]
    train_struct = mahalanobis(train_benign_emb, train_out["embedding"])
    cal_struct = mahalanobis(train_benign_emb, cal_out["embedding"])
    test_struct = mahalanobis(train_benign_emb, test_out["embedding"])
    train_novelty = 0.5 * ecdf(train_energy, train_energy) + 0.5 * ecdf(train_struct, train_struct)
    cal_novelty = 0.5 * ecdf(train_energy, cal_energy) + 0.5 * ecdf(train_struct, cal_struct)
    test_novelty = 0.5 * ecdf(train_energy, test_energy) + 0.5 * ecdf(train_struct, test_struct)

    # Disjoint known-only calibration scores set the conformal rejection rule.
    cal_p = (1.0 + (cal_novelty[None, :] >= cal_novelty[:, None]).sum(axis=1)) / (len(cal_novelty) + 1.0)
    test_p = (1.0 + (cal_novelty[None, :] >= test_novelty[:, None]).sum(axis=1)) / (len(cal_novelty) + 1.0)
    alpha = 0.05
    test_rejected = test_p <= alpha
    # Heteroscedastic clinical expectation trained jointly but loss-weighted only
    # on benign train claims; thresholds are from benign calibration claims.
    cal_mu, cal_log_sigma = cal_out["mu"], cal_out["log_sigma"]
    test_mu, test_log_sigma = test_out["mu"], test_out["log_sigma"]
    target_cal = targets[selection_end:calibration_end].detach().cpu().numpy()
    target_test = targets[calibration_end:n_samples].detach().cpu().numpy()
    cal_clinical = np.maximum(0.0, (target_cal - cal_mu) / np.exp(cal_log_sigma)).max(axis=1)
    test_clinical = np.maximum(0.0, (target_test - test_mu) / np.exp(test_log_sigma)).max(axis=1)
    cal_labels = labels[selection_end:calibration_end]
    cal_benign = cal_labels == 0
    tau_clinical = max(float(np.quantile(cal_clinical[cal_benign], 0.95)), 1e-6)
    tau_structural = max(float(np.quantile(cal_struct[cal_benign], 0.95)), 1e-6)
    unexplainedness = np.maximum(test_clinical / tau_clinical, test_struct / tau_structural)
    escalated = test_rejected & (unexplainedness > 1.0)

    test_labels = labels[calibration_end:n_samples]
    test_novel = test_labels == hidden_label
    test_known = ~test_novel
    logits = test_out["logits"]
    predicted_orig = class_to_label[np.argmax(logits, axis=1)]
    known_y = test_labels[test_known]
    known_pred = predicted_orig[test_known]
    metrics: dict[str, Any] = {
        "known_accuracy": float(accuracy_score(known_y, known_pred)) if len(known_y) else None,
        "known_macro_f1": float(f1_score(known_y, known_pred, labels=known_labels,
                                         average="macro", zero_division=0)) if len(known_y) else None,
        "novel_auroc": float(roc_auc_score(test_novel.astype(int), test_novelty))
                        if len(np.unique(test_novel)) == 2 else None,
        "novel_ap": float(average_precision_score(test_novel.astype(int), test_novelty))
                    if test_novel.any() else None,
        "novel_recall_at_alpha": float(test_rejected[test_novel].mean()) if test_novel.any() else None,
        "known_false_rejection_at_alpha": float(test_rejected[test_known].mean()) if test_known.any() else None,
        "rare_severe_benign_false_rejection": (
            float(test_rejected[(test_labels == 0) & data["benign_outlier"][calibration_end:n_samples]].mean())
            if np.any((test_labels == 0) & data["benign_outlier"][calibration_end:n_samples]) else None
        ),
        "gated_novel_recall": float(escalated[test_novel].mean()) if test_novel.any() else None,
        "escalation_rate": float(escalated.mean()),
        "calibration_known_false_rejection": float((cal_p <= alpha).mean()),
        "tau_clinical": tau_clinical,
        "tau_structural": tau_structural,
    }
    preds = []
    for j, idx in enumerate(range(calibration_end, n_samples)):
        preds.append({
            "claim_index": int(idx), "timestamp_day": float(data["day"][idx]),
            "true_mechanism": str(data["label_records"][idx]["mechanism"]),
            "is_heldout_mechanism": bool(test_novel[j]),
            "predicted_known_class": MECHANISMS[int(predicted_orig[j]) - 1]
                if predicted_orig[j] > 0 else "benign",
            "novelty_score": float(test_novelty[j]), "conformal_p": float(test_p[j]),
            "clinical_inconsistency": float(test_clinical[j]),
            "structural_deviation": float(test_struct[j]),
            "explained_rarity_escalation": bool(escalated[j]),
        })

    report = {
        "experiment": "OW-ERF temporal heterogeneous graph synthetic benchmark",
        "status": "synthetic research baseline; not validated on real claims",
        "seed": int(seed), "hidden_mechanism": hidden_mechanism, "tier": 2,
        "n_samples": int(n_samples), "epochs_requested": int(epochs),
        "best_epoch": int(best_epoch), "selection_loss": float(best_val),
        "train_claims": train_end, "selection_claims": selection_end - train_end,
        "calibration_claims": calibration_end - selection_end,
        "deployment_claims": n_samples - calibration_end,
        "known_original_labels": known_labels, "hidden_original_label": hidden_label,
        "alpha": alpha, "metrics": metrics, "training_history": history_log,
        "graph_schema": schema.schema_report(),
        "temporal_leakage_controls": [
            "Chronological training/selection/calibration/deployment split.",
            "Scaler fitting uses only the training window.",
            "Calibration uses the separate known-only calibration window.",
            "Each query block aggregates claim messages only from indices strictly before that block.",
            "Target-block claims never message into entity nodes used by claims in that same block.",
            "Labels and mechanism IDs are not node features or graph edges."
        ],
    }
    fold_dir = output_dir / f"seed_{seed}_{hidden_mechanism}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (fold_dir / "test_predictions.json").write_text(json.dumps(preds, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), fold_dir / "model_state.pt")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--hidden", default="all", help="One mechanism name or 'all'")
    parser.add_argument("--output-dir", default="results/temporal_heterograph")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.n_samples < 800:
        parser.error("--n-samples must be >= 800")
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    hidden_mechanisms = list(MECHANISMS) if args.hidden == "all" else [args.hidden]
    if any(item not in MECHANISMS for item in hidden_mechanisms):
        parser.error(f"--hidden must be 'all' or one of {MECHANISMS}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for hidden in hidden_mechanisms:
        for seed in seeds:
            print(f"[graph] seed={seed} hidden={hidden} n={args.n_samples} device={device}", flush=True)
            reports.append(train_fold(
                n_samples=args.n_samples, hidden_mechanism=hidden, seed=seed,
                epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                output_dir=output_dir, device=device
            ))
    summary = {
        "title": "Temporal heterogeneous graph OW-ERF synthetic benchmark",
        "model": "typed temporal relation-specific message passing + relation attention + calibrated explained-rarity gate",
        "protocol": {
            "folds": len(reports), "seeds": seeds, "hidden_mechanisms": hidden_mechanisms,
            "n_samples_per_fold": args.n_samples, "epoch_cap": args.epochs,
            "batch_size": args.batch_size, "device": str(device),
            "data": "parametric synthetic claims only; not real claim data",
        },
        "aggregate_metrics": {},
        "folds": [{"seed": r["seed"], "hidden_mechanism": r["hidden_mechanism"],
                   "best_epoch": r["best_epoch"], **r["metrics"]} for r in reports],
    }
    for metric in METRICS:
        values = [float(r["metrics"][metric]) for r in reports if r["metrics"].get(metric) is not None]
        summary["aggregate_metrics"][metric] = {
            "n_folds": len(values),
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
        }
    (output_dir / "temporal_heterograph_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    csv_lines = ["seed,hidden_mechanism,best_epoch," + ",".join(METRICS)]
    for row in summary["folds"]:
        csv_lines.append(",".join(str(row.get(key, "")) for key in
            ["seed", "hidden_mechanism", "best_epoch", *METRICS]))
    (output_dir / "temporal_heterograph_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
