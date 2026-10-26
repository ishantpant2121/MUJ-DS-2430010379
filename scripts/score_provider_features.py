#!/usr/bin/env python3
"""Score a provider feature table using the saved public-benchmark model."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def score_features(model_path: Path, feature_csv: Path, output_path: Path) -> None:
    bundle = joblib.load(model_path)
    frame = pd.read_csv(feature_csv)
    ids = frame["Provider"].astype(str) if "Provider" in frame else pd.Series([f"row_{i}" for i in range(len(frame))])
    selected = list(bundle["feature_columns"])
    X = frame.drop(columns=["Provider", "PotentialFraud", "target"], errors="ignore").copy()
    missing = [c for c in selected if c not in X.columns]
    if missing:
        raise ValueError(
            "Input feature CSV does not satisfy this model's feature contract. "
            f"Missing {len(missing)} required columns; first missing: {missing[:8]}"
        )
    X = X[selected].replace([np.inf, -np.inf], np.nan)
    probability = bundle["model"].predict_proba(X)[:, 1]
    threshold = float(bundle["threshold"])
    output = pd.DataFrame({
        "Provider": ids,
        "potential_fraud_risk_score": probability,
        "priority_flag_at_validation_calibrated_threshold": probability >= threshold,
        "threshold": threshold,
        "model": bundle["model_name"],
        "target_semantics": "provider-level PotentialFraud benchmark label; not claim-level proof",
    }).sort_values("potential_fraud_risk_score", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Scored {len(output)} providers; wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/public_provider_fraud_model.joblib")
    parser.add_argument("--features", default="results/public_provider_features.csv")
    parser.add_argument("--output", default="results/public_provider_scored_features.csv")
    args = parser.parse_args()
    score_features(Path(args.model), Path(args.features), Path(args.output))


if __name__ == "__main__":
    main()
