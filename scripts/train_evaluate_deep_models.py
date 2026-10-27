#!/usr/bin/env python3
"""Train and fairly evaluate standalone LSTM, 1D-CNN and Transformer baselines.

The three models are trained independently on the same synthetic generator,
seed, leave-one-mechanism-out fold, temporal-independent split and evaluation
protocol. This is a controlled synthetic benchmark; it is not real insurance
claim validation and does not replace the fused OW-ERF detector.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from owerf_deep_experiment import (
    CODE_VOCAB,
    MECHANISMS,
    _stratified_split,
    generate_synthetic_claims,
)

MODEL_NAMES = ("LSTM", "CNN1D", "Transformer")
ALPHA = 0.05


class LSTMClaimClassifier(nn.Module):
    """LSTM over an ordered window of recent provider/beneficiary claims."""
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 48, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor):
        _, (h_n, _) = self.encoder(x)
        embedding = self.dropout(self.norm(h_n[-1]))
        return embedding, self.classifier(embedding)


class CNNClaimClassifier(nn.Module):
    """1D-CNN over ordered claim procedure/line-item code sequences."""
    def __init__(self, vocab_size: int, n_classes: int, embedding_dim: int = 24, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, 48, kernel_size=3, padding=1),
            nn.Conv1d(48, 48, kernel_size=5, padding=2),
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(48), nn.BatchNorm1d(48)])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.LayerNorm(48))
        self.classifier = nn.Linear(48, n_classes)

    def forward(self, tokens: torch.Tensor):
        x = self.embedding(tokens).transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            x = self.dropout(F.relu(norm(conv(x))))
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        embedding = self.projection(pooled)
        return embedding, self.classifier(embedding)


class TransformerClaimClassifier(nn.Module):
    """Transformer encoder over claim/hospital/doctor/package/diagnosis context tokens."""
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 48, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor):
        x = self.input_projection(tokens)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.zeros((padding_mask.shape[0], 1), dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)
        encoded = self.encoder(x, src_key_padding_mask=full_mask)
        embedding = self.dropout(encoded[:, 0])
        return embedding, self.classifier(embedding)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def _make_model(name: str, n_classes: int, input_dims: dict[str, int], device: torch.device) -> nn.Module:
    if name == "LSTM":
        model = LSTMClaimClassifier(input_dims["history"], n_classes)
    elif name == "CNN1D":
        model = CNNClaimClassifier(CODE_VOCAB, n_classes)
    elif name == "Transformer":
        model = TransformerClaimClassifier(input_dims["context"], n_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)


def _prepare_inputs(
    name: str,
    data: dict[str, np.ndarray],
    train_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit any numeric scaling only on training rows; CNN consumes categorical code IDs."""
    if name == "LSTM":
        scaler = StandardScaler().fit(data["hist"][train_ids].reshape(-1, data["hist"].shape[-1]))
        transformed = scaler.transform(data["hist"].reshape(-1, data["hist"].shape[-1]))
        x = transformed.reshape(data["hist"].shape).astype(np.float32)
        return {"x": x}, {
            "input_type": "history_sequence",
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
    if name == "CNN1D":
        return {"x": data["codes"].astype(np.int64)}, {
            "input_type": "ordered_procedure_code_sequence",
            "vocabulary_size": int(CODE_VOCAB),
            "embedding_dim": 24,
        }
    if name == "Transformer":
        fit_tokens = data["ctx"][train_ids]
        fit_mask = data["ctx_pad_mask"][train_ids]
        valid_tokens = fit_tokens[~fit_mask]
        scaler = StandardScaler().fit(valid_tokens)
        flat = scaler.transform(data["ctx"].reshape(-1, data["ctx"].shape[-1]))
        x = flat.reshape(data["ctx"].shape).astype(np.float32)
        return {"x": x, "padding_mask": data["ctx_pad_mask"].astype(bool)}, {
            "input_type": "relational_context_tokens",
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "padding_mask_used": True,
        }
    raise ValueError(f"Unknown model: {name}")


def _forward_batch(
    model: nn.Module, name: str, x: np.ndarray, ids: np.ndarray,
    device: torch.device, padding_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ids), 256):
            batch_ids = ids[start:start + 256]
            xb = torch.as_tensor(x[batch_ids], device=device)
            if name == "CNN1D":
                xb = xb.long()
            else:
                xb = xb.float()
            if name == "Transformer":
                mask = torch.as_tensor(padding_mask[batch_ids], device=device, dtype=torch.bool)
                emb, logits = model(xb, mask)
            else:
                emb, logits = model(xb)
            embeddings.append(emb.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(logits_list)


def _energy(logits: np.ndarray) -> np.ndarray:
    row_max = np.max(logits, axis=1, keepdims=True)
    return (-row_max[:, 0] - np.log(np.exp(logits - row_max).sum(axis=1))).astype(float)


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(reference, values, side="right") / (len(reference) + 1.0)


def _class_mahalanobis(train_embeddings: np.ndarray, train_y: np.ndarray, query: np.ndarray, n_classes: int) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(train_embeddings, rowvar=False))
    covariance = covariance + np.eye(covariance.shape[0]) * 0.1
    precision = np.linalg.pinv(covariance)
    distances = []
    for class_id in range(n_classes):
        members = train_embeddings[train_y == class_id]
        if len(members) == 0:
            raise RuntimeError(f"Missing class {class_id} in model training data.")
        center = members.mean(axis=0)
        delta = query - center
        distances.append(np.maximum(0.0, np.einsum("bi,ij,bj->b", delta, precision, delta)))
    return np.min(np.stack(distances, axis=1), axis=1)


def _fit_one_model(
    name: str,
    data: dict[str, np.ndarray],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    cal_ids: np.ndarray,
    test_ids: np.ndarray,
    y_mapped: np.ndarray,
    known_original: list[int],
    hidden_id: int,
    epochs: int,
    batch_size: int,
    seed: int,
    output_dir: Path,
    save_weights: bool,
) -> dict[str, Any]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared, preprocessing = _prepare_inputs(name, data, train_ids)
    x = prepared["x"]
    pad_mask = prepared.get("padding_mask")
    model = _make_model(name, len(known_original), {
        "history": data["hist"].shape[-1],
        "context": data["ctx"].shape[-1],
    }, device)

    class_counts = np.bincount(y_mapped[train_ids], minlength=len(known_original)).astype(float)
    if np.any(class_counts == 0):
        raise RuntimeError("Training split is missing a known class.")
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 9)
    train_loss_history: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_ids)
        losses = []
        for start in range(0, len(shuffled), batch_size):
            ids = shuffled[start:start + batch_size]
            if len(ids) < 8:
                continue
            xb = torch.as_tensor(x[ids], device=device)
            xb = xb.long() if name == "CNN1D" else xb.float()
            yb = torch.as_tensor(y_mapped[ids], dtype=torch.long, device=device)
            if name == "Transformer":
                mask = torch.as_tensor(pad_mask[ids], dtype=torch.bool, device=device)
                _, logits = model(xb, mask)
            else:
                _, logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weights_tensor)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} produced non-finite loss.")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise RuntimeError(f"No {name} training batches were processed.")
        train_loss_history.append(float(np.mean(losses)))
        _, val_logits = _forward_batch(model, name, x, val_ids, device, pad_mask)
        val_loss = float(F.cross_entropy(
            torch.as_tensor(val_logits, dtype=torch.float32),
            torch.as_tensor(y_mapped[val_ids], dtype=torch.long),
        ).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No valid {name} checkpoint was selected.")
    model.load_state_dict(best_state)
    train_embeddings, train_logits = _forward_batch(model, name, x, train_ids, device, pad_mask)
    val_embeddings, val_logits = _forward_batch(model, name, x, val_ids, device, pad_mask)
    cal_embeddings, cal_logits = _forward_batch(model, name, x, cal_ids, device, pad_mask)
    test_embeddings, test_logits = _forward_batch(model, name, x, test_ids, device, pad_mask)

    train_y = y_mapped[train_ids]
    energy_val, energy_cal, energy_test = _energy(val_logits), _energy(cal_logits), _energy(test_logits)
    md_val = _class_mahalanobis(train_embeddings, train_y, val_embeddings, len(known_original))
    md_cal = _class_mahalanobis(train_embeddings, train_y, cal_embeddings, len(known_original))
    md_test = _class_mahalanobis(train_embeddings, train_y, test_embeddings, len(known_original))
    # Fit both empirical CDFs on a disjoint validation window, then apply that same
    # frozen transform to calibration and test windows.
    val_score = 0.5 * _ecdf(energy_val, energy_val) + 0.5 * _ecdf(md_val, md_val)
    cal_score = 0.5 * _ecdf(energy_val, energy_cal) + 0.5 * _ecdf(md_val, md_cal)
    test_score = 0.5 * _ecdf(energy_val, energy_test) + 0.5 * _ecdf(md_val, md_test)
    cal_p = (1 + (cal_score[:, None] >= test_score[None, :]).sum(axis=0)) / (len(cal_score) + 1.0)
    novel_flags = cal_p <= ALPHA

    test_original_labels = data["labels"][test_ids]
    is_novel = test_original_labels == hidden_id
    is_known = ~is_novel
    known_test = np.flatnonzero(is_known)
    y_true = y_mapped[test_ids][known_test]
    y_pred = test_logits.argmax(axis=1)[known_test]
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    known_class_report = classification_report(
        y_true, y_pred, labels=list(range(len(known_original))),
        target_names=[("benign" if k == 0 else MECHANISMS[k - 1]) for k in known_original],
        output_dict=True, zero_division=0,
    )
    test_original_prediction = np.asarray(known_original)[test_logits.argmax(axis=1)]
    binary_y = (test_original_labels[is_known] != 0).astype(int)
    binary_pred = (test_original_prediction[is_known] != 0).astype(int)
    binary_metrics = {
        "accuracy": float(accuracy_score(binary_y, binary_pred)),
        "precision": float(precision_score(binary_y, binary_pred, zero_division=0)),
        "recall": float(recall_score(binary_y, binary_pred, zero_division=0)),
        "f1": float(f1_score(binary_y, binary_pred, zero_division=0)),
    }

    novelty_auc = float(roc_auc_score(is_novel.astype(int), test_score))
    novelty_ap = float(average_precision_score(is_novel.astype(int), test_score))
    fpr_curve, tpr_curve, thresholds = roc_curve(is_novel.astype(int), test_score)
    eligible = np.flatnonzero(tpr_curve >= 0.95)
    fpr_at_95_tpr = float(fpr_curve[eligible[0]]) if len(eligible) else 1.0
    known_false_rejection = float(novel_flags[is_known].mean()) if is_known.any() else None
    novel_recall = float(novel_flags[is_novel].mean()) if is_novel.any() else None
    rare_severe = data["rare_severe"][test_ids]
    rare_severe_false_rejection = float(novel_flags[rare_severe].mean()) if rare_severe.any() else None
    k = min(100, len(test_ids))
    top = np.argsort(test_score)[::-1][:k]
    p_at_k = float(is_novel[top].mean()) if k else None

    save_path = None
    if save_weights:
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        save_path = weights_dir / f"{name.lower()}_hidden_{MECHANISMS[hidden_id - 1]}_seed_{seed}.pt"
        checkpoint = {
            "model_name": name,
            "hidden_mechanism": MECHANISMS[hidden_id - 1],
            "known_original_labels": known_original,
            "state_dict": best_state,
            "preprocessing": preprocessing,
            "model_config": {
                "n_classes": len(known_original),
                "history_dim": int(data["hist"].shape[-1]),
                "context_dim": int(data["ctx"].shape[-1]),
                "code_vocab": int(CODE_VOCAB),
                "best_epoch": best_epoch,
                "seed": seed,
            },
            "validation_energy": energy_val.astype(np.float32),
            "validation_mahalanobis": md_val.astype(np.float32),
            "calibration_score": cal_score.astype(np.float32),
            "alpha": ALPHA,
        }
        torch.save(checkpoint, save_path)

    return {
        "model": name,
        "hidden_mechanism": MECHANISMS[hidden_id - 1],
        "seed": seed,
        "device": str(device),
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "train_loss_first_epoch": train_loss_history[0],
        "train_loss_best_epoch": train_loss_history[best_epoch - 1],
        "validation_cross_entropy_at_best_epoch": best_val_loss,
        "n_train": int(len(train_ids)),
        "n_validation": int(len(val_ids)),
        "n_calibration": int(len(cal_ids)),
        "n_test": int(len(test_ids)),
        "n_test_known": int(len(known_test)),
        "n_test_novel": int(is_novel.sum()),
        "known_class_accuracy": accuracy,
        "known_class_macro_f1": macro_f1,
        "known_class_balanced_accuracy": balanced_accuracy,
        "known_class_report": known_class_report,
        "known_test_fraud_vs_benign_metrics": binary_metrics,
        "novel_mechanism_auroc": novelty_auc,
        "novel_mechanism_average_precision": novelty_ap,
        "false_positive_rate_at_95pct_novel_recall": fpr_at_95_tpr,
        "novel_recall_at_conformal_alpha_005": novel_recall,
        "known_false_rejection_rate_at_conformal_alpha_005": known_false_rejection,
        "rare_severe_benign_false_rejection_rate": rare_severe_false_rejection,
        "precision_at_k": {"k": k, "value": p_at_k},
        "conformal_alpha": ALPHA,
        "novelty_method": "equal-weight energy empirical-CDF + latent Mahalanobis empirical-CDF; split conformal calibration",
        "input_modality": preprocessing["input_type"],
        "trained_weights_path": str(save_path.relative_to(output_dir)) if save_path else None,
    }


def run_benchmark(
    n_samples: int = 2400,
    epochs: int = 5,
    batch_size: int = 128,
    seeds: tuple[int, ...] = (42, 43, 44),
    output_dir: str = "results/standalone_deep_models",
    save_weights: bool = True,
    hidden_mechanisms: tuple[str, ...] = MECHANISMS,
) -> dict[str, Any]:
    if n_samples < 1000:
        raise ValueError("Use at least 1000 claims for stable split coverage.")
    if epochs < 1 or batch_size < 8:
        raise ValueError("epochs must be >= 1 and batch_size must be >= 8.")
    if not seeds:
        raise ValueError("At least one random seed is required.")
    if not hidden_mechanisms or any(m not in MECHANISMS for m in hidden_mechanisms):
        raise ValueError(f"hidden_mechanisms must be a non-empty subset of {MECHANISMS}.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        data = generate_synthetic_claims(n_samples=n_samples, seed=seed)
        train_all, calibration_all, test_ids = _stratified_split(data["labels"], seed + 1)
        for hidden in hidden_mechanisms:
            hidden_id = MECHANISMS.index(hidden) + 1
            known_original = [0] + [i for i in range(1, len(MECHANISMS) + 1) if i != hidden_id]
            mapped = np.full(n_samples, -1, dtype=np.int64)
            original_to_mapped = {original: mapped_value for mapped_value, original in enumerate(known_original)}
            for original, mapped_value in original_to_mapped.items():
                mapped[data["labels"] == original] = mapped_value

            train_pool = train_all[data["labels"][train_all] != hidden_id]
            # The selection window is carved only from the training partition.
            train_ids, val_ids = train_test_split(
                train_pool, test_size=0.15, random_state=seed + hidden_id,
                stratify=mapped[train_pool],
            )
            cal_ids = calibration_all[data["labels"][calibration_all] != hidden_id]
            if len(cal_ids) < 20:
                raise RuntimeError("Too few known-only calibration claims.")
            if len(np.unique(mapped[train_ids])) != len(known_original):
                raise RuntimeError("A known class is missing from this training partition.")
            if not np.any(data["labels"][test_ids] == hidden_id):
                raise RuntimeError(f"Test fold {hidden} has no hidden-mechanism claims.")

            for model_index, name in enumerate(MODEL_NAMES):
                result = _fit_one_model(
                    name=name,
                    data=data,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    cal_ids=cal_ids,
                    test_ids=test_ids,
                    y_mapped=mapped,
                    known_original=known_original,
                    hidden_id=hidden_id,
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=seed + model_index * 100,
                    output_dir=output,
                    save_weights=save_weights,
                )
                result["data_seed"] = seed
                result["train_validation_calibration_protocol"] = "stratified train pool; 15% known-only validation; separate known-only calibration window; untouched held-out test partition"
                results.append(result)

    summary: dict[str, Any] = {}
    numeric_keys = [
        "known_class_accuracy", "known_class_macro_f1", "known_class_balanced_accuracy",
        "novel_mechanism_auroc", "novel_mechanism_average_precision",
        "false_positive_rate_at_95pct_novel_recall",
        "novel_recall_at_conformal_alpha_005",
        "known_false_rejection_rate_at_conformal_alpha_005",
        "rare_severe_benign_false_rejection_rate",
    ]
    for name in MODEL_NAMES:
        subset = [row for row in results if row["model"] == name]
        summary[name] = {}
        for metric in numeric_keys:
            values = [float(row[metric]) for row in subset if row[metric] is not None and np.isfinite(row[metric])]
            summary[name][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
                "n_runs": len(values),
            }
    report = {
        "experiment": "standalone LSTM vs 1D-CNN vs Transformer under leave-one-mechanism-out",
        "scope": "synthetic data only; no real patient or insurance claim data",
        "purpose": "separately train and evaluate each architecture on the same multimodal synthetic benchmark",
        "metrics_semantics": {
            "known_class_accuracy": "multiclass accuracy on test claims from classes seen during training; held-out mechanism excluded from this metric",
            "novel_mechanism_auroc": "held-out fraud mechanism positive vs every non-held-out test claim, using fused energy/Mahalanobis novelty",
            "novel_recall_at_conformal_alpha_005": "fraction of held-out mechanism claims rejected by the calibration-set conformal p-value rule at alpha=0.05",
            "rare_severe_benign_false_rejection_rate": "fraction of rare-severe legitimate test claims rejected by the same novelty gate",
        },
        "configuration": {
            "n_samples_per_dataset_seed": n_samples,
            "epochs": epochs,
            "batch_size": batch_size,
            "seeds": list(seeds),
            "hidden_mechanisms": list(hidden_mechanisms),
            "models": list(MODEL_NAMES),
            "alpha": ALPHA,
        },
        "split_protocol": "same generator seed and stratified train/calibration/test partition across model architectures; hidden mechanism excluded from train, validation and calibration",
        "summary_by_model": summary,
        "runs": results,
        "limitations": [
            "Synthetic generator contains designed proxies for each mechanism; metrics measure performance on this generator only.",
            "A held-out mechanism is evaluated as novelty; it is not a separately supervised output class.",
            "The three architectures are trained independently, not as the same fused encoder, so the comparison isolates their input modality/architecture choices but modalities differ as intended by the paper.",
            "Temporal graph encoder and public provider-level benchmark are separate experiments and not combined into this standalone comparison.",
            "Three seeds provide an initial stability check, not publication-grade confidence intervals; complete the registered seed/ablation plan before submission.",
        ],
    }
    report_path = output / "standalone_deep_model_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # Compact CSV with one row per architecture / held-out mechanism / seed.
    import csv
    fieldnames = [
        "model", "hidden_mechanism", "data_seed", "best_epoch",
        "known_class_accuracy", "known_class_macro_f1", "known_class_balanced_accuracy",
        "known_test_fraud_vs_benign_metrics", "novel_mechanism_auroc",
        "novel_mechanism_average_precision", "false_positive_rate_at_95pct_novel_recall",
        "novel_recall_at_conformal_alpha_005", "known_false_rejection_rate_at_conformal_alpha_005",
        "rare_severe_benign_false_rejection_rate", "precision_at_k", "trained_weights_path",
    ]
    with (output / "standalone_deep_model_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fieldnames})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", default="42,43,44", help="comma-separated generator seeds")
    parser.add_argument("--hidden", default="all", help="one mechanism, comma-separated mechanisms, or 'all'")
    parser.add_argument("--output-dir", default="results/standalone_deep_models")
    parser.add_argument("--no-save-weights", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    hidden = MECHANISMS if args.hidden == "all" else tuple(x.strip() for x in args.hidden.split(",") if x.strip())
    report = run_benchmark(
        n_samples=args.n_samples, epochs=args.epochs, batch_size=args.batch_size,
        seeds=seeds, output_dir=args.output_dir, save_weights=not args.no_save_weights,
        hidden_mechanisms=hidden,
    )
    print(json.dumps({"summary_by_model": report["summary_by_model"], "report_path": str(Path(args.output_dir) / "standalone_deep_model_metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
