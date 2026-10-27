#!/usr/bin/env python3
"""Build a side-by-side report for native-modality and controlled-input LSTM/CNN/Transformer runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

METRICS = (
    "known_class_accuracy",
    "known_class_macro_f1",
    "known_class_balanced_accuracy",
    "novel_mechanism_auroc",
    "novel_mechanism_average_precision",
    "novel_recall_at_conformal_alpha_005",
    "known_false_rejection_rate_at_conformal_alpha_005",
    "rare_severe_benign_false_rejection_rate",
)
MODELS = ("LSTM", "CNN1D", "Transformer")


def _read_report(path: Path, expected_mode: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("input_mode") != expected_mode:
        raise ValueError(f"{path} declares input_mode={report.get('input_mode')!r}; expected {expected_mode!r}")
    runs = report.get("runs", [])
    expected = {(model, hidden, seed) for model in MODELS
                for hidden in report["configuration"]["hidden_mechanisms"]
                for seed in report["configuration"]["seeds"]}
    observed = {(row.get("model"), row.get("hidden_mechanism"), row.get("data_seed")) for row in runs}
    if observed != expected:
        raise ValueError(f"{path} run coverage mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    return report


def build_comparison(native_path: Path, shared_path: Path) -> dict[str, Any]:
    native = _read_report(native_path, "native")
    shared = _read_report(shared_path, "shared_context")
    if native["configuration"]["seeds"] != shared["configuration"]["seeds"]:
        raise ValueError("Native and shared-context runs must use identical generator seeds.")
    if native["configuration"]["hidden_mechanisms"] != shared["configuration"]["hidden_mechanisms"]:
        raise ValueError("Native and shared-context runs must use identical held-out mechanisms.")
    if native["configuration"]["n_samples_per_dataset_seed"] != shared["configuration"]["n_samples_per_dataset_seed"]:
        raise ValueError("Native and shared-context runs must use the same sample count.")
    if native["configuration"]["epochs"] != shared["configuration"]["epochs"]:
        raise ValueError("Native and shared-context runs must use the same epoch budget.")

    rows: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for model in MODELS:
        n = native["summary_by_model"][model]
        s = shared["summary_by_model"][model]
        for metric in METRICS:
            nm, sm = n[metric]["mean"], s[metric]["mean"]
            rows.append({
                "model": model, "metric": metric,
                "native_mean": nm, "native_std": n[metric]["std"], "native_n": n[metric]["n_runs"],
                "shared_context_mean": sm, "shared_context_std": s[metric]["std"], "shared_context_n": s[metric]["n_runs"],
                "delta_shared_minus_native": (sm - nm) if nm is not None and sm is not None else None,
            })
            deltas.append({
                "model": model, "metric": metric,
                "native_mean": nm, "shared_context_mean": sm,
                "delta_shared_minus_native": (sm - nm) if nm is not None and sm is not None else None,
            })
    return {
        "experiment": "controlled LSTM vs CNN1D vs Transformer comparison",
        "scope": "synthetic data only; no real patient or insurance claim data",
        "design": "Paired generator seeds and held-out-mechanism folds. Native mode uses each architecture's intended modality; shared_context mode supplies identical standardized context tokens and padding masks to all three. Preprocessing is fitted on training rows only, model checkpoint selection uses validation loss, calibration is disjoint, and held-out test data are used once for final metrics.",
        "configuration": {
            "n_samples_per_dataset_seed": native["configuration"]["n_samples_per_dataset_seed"],
            "epochs": native["configuration"]["epochs"],
            "batch_size": native["configuration"]["batch_size"],
            "seeds": native["configuration"]["seeds"],
            "hidden_mechanisms": native["configuration"]["hidden_mechanisms"],
            "alpha": native["configuration"]["alpha"],
            "runs_per_mode": len(native["runs"]),
            "total_model_fits": len(native["runs"]) + len(shared["runs"]),
        },
        "summary_rows": rows,
        "interpretation_guardrails": [
            "Native comparison confounds architecture with input modality.",
            "Shared-context comparison controls the input representation and is the appropriate architecture ablation, but uses a synthetic context representation rather than each modality's natural input.",
            "All results are conditional on the synthetic generator; none establishes real-claims performance.",
            "Small seed counts produce descriptive means/standard deviations, not a confirmatory statistical claim.",
        ],
        "native_report": str(native_path.as_posix()),
        "shared_context_report": str(shared_path.as_posix()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--shared-context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_comparison(args.native, args.shared_context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["summary_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(report["summary_rows"])
    print(json.dumps({
        "output": str(args.output),
        "csv_output": str(csv_path),
        "total_model_fits": report["configuration"]["total_model_fits"],
        "run_count_per_mode": report["configuration"]["runs_per_mode"],
    }, indent=2))


if __name__ == "__main__":
    main()
