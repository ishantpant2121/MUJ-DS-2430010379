#!/usr/bin/env python3
"""Train/test a provider-level Medicare fraud-risk baseline on a public dataset.

This experiment is separate from OW-ERF's synthetic claim-level mechanism tests:
the source labels PotentialFraud at provider level and does not label individual
claims as upcoding, unbundling, phantom billing, or collusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY = "https://github.com/reshmanalage/healthcare-provider-fraud-analytics"
DATA_SOURCE = "https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis"
DATASET_MENDELEY = "https://data.mendeley.com/datasets/gsn2hyty37/1"
RAW_BASE = "https://raw.githubusercontent.com/reshmanalage/healthcare-provider-fraud-analytics/main/data/raw"
FILES = {
    "labels": "train_labels.csv",
    "beneficiary": "train_beneficiary.csv",
    "inpatient": "train_inpatient.csv",
    "outpatient": "train_outpatient.csv",
}
SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset(data_dir: Path, force_download: bool = False) -> dict[str, str]:
    """Download the four labelled-train tables; raw data stays outside Git."""
    data_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for logical_name, filename in FILES.items():
        path = data_dir / filename
        if force_download or not path.exists() or path.stat().st_size == 0:
            url = f"{RAW_BASE}/{filename}"
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "OW-ERF-reproducible-research-prototype/1.0"},
            )
            temporary = path.with_suffix(path.suffix + ".download")
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                if temporary.stat().st_size < 100:
                    raise RuntimeError(f"Downloaded file looks empty: {url}")
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        hashes[filename] = _sha256(path)
    return hashes


def _columns(path: Path) -> list[str]:
    return [str(col).strip() for col in pd.read_csv(path, nrows=0).columns]


def _read_claims(path: Path) -> pd.DataFrame:
    """Load only fields used by aggregation, reducing memory for the 558K-claim table."""
    cols = _columns(path)
    keep = {
        "Provider", "ClaimID", "BeneID",
        "InscClaimAmtReimbursed", "DeductibleAmtPaid",
        "AttendingPhysician", "OperatingPhysician", "OtherPhysician",
        "ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt",
        "DiagnosisGroupCode",
    }
    keep.update(c for c in cols if c.startswith("ClmDiagnosisCode_"))
    keep.update(c for c in cols if c.startswith("ClmProcedureCode_"))
    usecols = [col for col in cols if col in keep]
    if "Provider" not in usecols:
        raise ValueError(f"Required Provider column not found in {path.name}")
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def _read_beneficiaries(path: Path) -> pd.DataFrame:
    cols = _columns(path)
    wanted = {"BeneID", "DOB", "Gender", "Race", "State", "County", "RenalDiseaseIndicator"}
    wanted.update(col for col in cols if col.startswith("ChronicCond_"))
    frame = pd.read_csv(path, usecols=[c for c in cols if c in wanted], low_memory=False)
    frame.columns = [str(col).strip() for col in frame.columns]
    if "BeneID" not in frame:
        raise ValueError("Beneficiary table has no BeneID column.")
    frame = frame.drop_duplicates("BeneID", keep="first").copy()
    if "DOB" in frame:
        frame["DOB"] = pd.to_datetime(frame["DOB"], errors="coerce")
    for col in [c for c in frame if c.startswith("ChronicCond_")]:
        frame[f"bene_{col}"] = (pd.to_numeric(frame[col], errors="coerce") == 1).astype(float)
        frame.drop(columns=col, inplace=True)
    if "RenalDiseaseIndicator" in frame:
        frame["bene_renal_disease"] = (
            frame["RenalDiseaseIndicator"].astype(str).str.upper().str.strip() == "Y"
        ).astype(float)
        frame.drop(columns="RenalDiseaseIndicator", inplace=True)
    if "Gender" in frame:
        for val in (1, 2):
            frame[f"bene_gender_{val}_share"] = (pd.to_numeric(frame["Gender"], errors="coerce") == val).astype(float)
        frame.drop(columns="Gender", inplace=True)
    if "Race" in frame:
        for val in (1, 2, 3, 4, 5):
            frame[f"bene_race_{val}_share"] = (pd.to_numeric(frame["Race"], errors="coerce") == val).astype(float)
        frame.drop(columns="Race", inplace=True)
    # Raw State and County codes are categorical; use their diversity, not numeric magnitude.
    return frame


def _aggregate_claim_table(
    frame: pd.DataFrame,
    kind: str,
    beneficiary: pd.DataFrame | None,
) -> pd.DataFrame:
    """Aggregate claim records into one row per provider without using the target label."""
    df = frame.copy()
    df["Provider"] = df["Provider"].astype(str).str.strip()
    df = df[df["Provider"].notna() & (df["Provider"] != "")].copy()
    prefix = f"{kind}_"
    if "InscClaimAmtReimbursed" in df:
        df["InscClaimAmtReimbursed"] = pd.to_numeric(df["InscClaimAmtReimbursed"], errors="coerce")
    if "DeductibleAmtPaid" in df:
        df["DeductibleAmtPaid"] = pd.to_numeric(df["DeductibleAmtPaid"], errors="coerce")
        df[f"{prefix}deductible_zero"] = (df["DeductibleAmtPaid"].fillna(0) == 0).astype(float)

    date_cols = ("ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt")
    for col in date_cols:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "ClaimStartDt" in df:
        df[f"{prefix}claim_month"] = df["ClaimStartDt"].dt.to_period("M").astype(str)
    if "AdmissionDt" in df and "DischargeDt" in df:
        los = (df["DischargeDt"] - df["AdmissionDt"]).dt.days
    else:
        los = pd.Series(np.nan, index=df.index)
    if "ClaimStartDt" in df and "ClaimEndDt" in df:
        fallback = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days
        los = los.where(los.notna(), fallback)
    df[f"{prefix}length_of_stay_days"] = los.where((los >= 0) & (los <= 365), np.nan)

    dx_cols = sorted(c for c in df if c.startswith("ClmDiagnosisCode_"))
    proc_cols = sorted(c for c in df if c.startswith("ClmProcedureCode_"))
    if dx_cols:
        df[f"{prefix}diagnosis_codes_per_claim"] = df[dx_cols].notna().sum(axis=1).astype(float)
    else:
        df[f"{prefix}diagnosis_codes_per_claim"] = 0.0
    if proc_cols:
        df[f"{prefix}procedure_codes_per_claim"] = df[proc_cols].notna().sum(axis=1).astype(float)
    else:
        df[f"{prefix}procedure_codes_per_claim"] = 0.0

    # Aggregate beneficiary composition of the provider's claim panel.
    if beneficiary is not None and "BeneID" in df and "BeneID" in beneficiary:
        join_cols = [c for c in beneficiary.columns if c not in {"State", "County"}]
        join_cols += [c for c in ("State", "County") if c in beneficiary.columns]
        join_cols = list(dict.fromkeys(join_cols))
        df = df.merge(beneficiary[join_cols], on="BeneID", how="left", validate="many_to_one")
        if "DOB" in df and "ClaimStartDt" in df:
            age = (df["ClaimStartDt"] - df["DOB"]).dt.days / 365.2425
            df[f"{prefix}beneficiary_age"] = age.where((age >= 0) & (age <= 115), np.nan)
        if "State" in df:
            df[f"{prefix}beneficiary_state"] = df["State"]
        if "County" in df:
            df[f"{prefix}beneficiary_county"] = df["County"]

    # Group-by aggregations. Provider identifier is only the grouping key and is not a feature.
    grouped = df.groupby("Provider", sort=False)
    out = grouped.size().rename(f"{prefix}claim_count").to_frame()
    metrics: dict[str, tuple[str, Any]] = {}
    if "ClaimID" in df:
        metrics[f"{prefix}unique_claim_ids"] = ("ClaimID", "nunique")
    if "BeneID" in df:
        metrics[f"{prefix}unique_beneficiaries"] = ("BeneID", "nunique")
    if "InscClaimAmtReimbursed" in df:
        for stat in ("sum", "mean", "median", "max", "std"):
            metrics[f"{prefix}reimbursement_{stat}"] = ("InscClaimAmtReimbursed", stat)
    if "DeductibleAmtPaid" in df:
        for stat in ("sum", "mean"):
            metrics[f"{prefix}deductible_{stat}"] = ("DeductibleAmtPaid", stat)
        metrics[f"{prefix}deductible_zero_share"] = (f"{prefix}deductible_zero", "mean")
    for col in ("AttendingPhysician", "OperatingPhysician", "OtherPhysician"):
        if col in df:
            short = {"AttendingPhysician": "attending", "OperatingPhysician": "operating", "OtherPhysician": "other"}[col]
            metrics[f"{prefix}{short}_physicians_nunique"] = (col, "nunique")
            metrics[f"{prefix}{short}_physician_missing_share"] = (col, lambda values: values.isna().mean())
    for col, short, stat in (
        (f"{prefix}length_of_stay_days", "los_mean", "mean"),
        (f"{prefix}length_of_stay_days", "los_median", "median"),
        (f"{prefix}length_of_stay_days", "los_max", "max"),
        (f"{prefix}diagnosis_codes_per_claim", "dx_codes_mean", "mean"),
        (f"{prefix}diagnosis_codes_per_claim", "dx_codes_max", "max"),
        (f"{prefix}procedure_codes_per_claim", "procedure_codes_mean", "mean"),
        (f"{prefix}procedure_codes_per_claim", "procedure_codes_max", "max"),
    ):
        metrics[f"{prefix}{short}"] = (col, stat)
    if f"{prefix}claim_month" in df:
        metrics[f"{prefix}active_months"] = (f"{prefix}claim_month", "nunique")
    for col, short in (
        ("ClmDiagnosisCode_1", "primary_diagnosis"),
        ("ClmProcedureCode_1", "primary_procedure"),
        ("DiagnosisGroupCode", "diagnosis_group"),
    ):
        if col in df:
            metrics[f"{prefix}{short}_nunique"] = (col, "nunique")
    for col in (f"{prefix}beneficiary_age",):
        if col in df:
            metrics[f"{prefix}beneficiary_age_mean"] = (col, "mean")
            metrics[f"{prefix}beneficiary_age_std"] = (col, "std")
    for col in (c for c in df if c.startswith("bene_")):
        metrics[f"{prefix}{col}_share"] = (col, "mean")
    for col in (f"{prefix}beneficiary_state", f"{prefix}beneficiary_county"):
        if col in df:
            metrics[f"{prefix}{col.rsplit('_', 1)[-1]}_nunique"] = (col, "nunique")

    if metrics:
        out = out.join(grouped.agg(**metrics), how="left")
    return out


def engineer_provider_features(
    inpatient: pd.DataFrame,
    outpatient: pd.DataFrame,
    beneficiaries: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create provider aggregates and provider-level target; no provider ID is used as a feature."""
    labels = labels.copy()
    labels.columns = [str(c).strip() for c in labels.columns]
    if not {"Provider", "PotentialFraud"}.issubset(labels.columns):
        raise ValueError("Labels must contain Provider and PotentialFraud columns.")
    labels["Provider"] = labels["Provider"].astype(str).str.strip()
    labels["target"] = labels["PotentialFraud"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    labels = labels.dropna(subset=["target"]).drop_duplicates("Provider", keep="first")
    bene = beneficiaries.copy()
    bene.columns = [str(c).strip() for c in bene.columns]
    bene = _prepare_beneficiary_frame(bene)

    ip = _aggregate_claim_table(inpatient, "ip", bene)
    op = _aggregate_claim_table(outpatient, "op", bene)
    features = ip.join(op, how="outer").replace([np.inf, -np.inf], np.nan)
    features = features.reindex(labels["Provider"].tolist()).copy()
    features.index.name = "Provider"
    # All model input columns are numeric; provider IDs stay outside the feature matrix.
    features = features.apply(pd.to_numeric, errors="coerce")
    keep = [c for c in features.columns if features[c].notna().any() and features[c].nunique(dropna=True) > 1]
    features = features[keep]
    y = labels.set_index("Provider")["target"].astype(int).reindex(features.index)
    return features, y, pd.Series(features.index.astype(str), index=features.index, name="Provider")


def _prepare_beneficiary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create explicit one-hot shares for binary/categorical fields and chronic-condition flags."""
    bene = frame.copy()
    if "BeneID" not in bene:
        return pd.DataFrame(columns=["BeneID"])
    bene = bene.drop_duplicates("BeneID", keep="first")
    if "DOB" in bene:
        bene["DOB"] = pd.to_datetime(bene["DOB"], errors="coerce")
    for col in [c for c in bene if c.startswith("ChronicCond_")]:
        bene[f"bene_{col}"] = (pd.to_numeric(bene[col], errors="coerce") == 1).astype(float)
        bene.drop(columns=col, inplace=True)
    if "RenalDiseaseIndicator" in bene:
        bene["bene_renal_disease"] = (bene["RenalDiseaseIndicator"].astype(str).str.upper().str.strip() == "Y").astype(float)
        bene.drop(columns="RenalDiseaseIndicator", inplace=True)
    if "Gender" in bene:
        for value in (1, 2):
            bene[f"bene_gender_{value}"] = (pd.to_numeric(bene["Gender"], errors="coerce") == value).astype(float)
        bene.drop(columns="Gender", inplace=True)
    if "Race" in bene:
        for value in (1, 2, 3, 4, 5):
            bene[f"bene_race_{value}"] = (pd.to_numeric(bene["Race"], errors="coerce") == value).astype(float)
        bene.drop(columns="Race", inplace=True)
    return bene


def _metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    predicted = score >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, predicted, labels=[0, 1]).ravel())
    return {
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, score)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "alert_rate": float(predicted.mean()),
        "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
    }


def train_test_provider_model(
    features: pd.DataFrame,
    labels: pd.Series,
    provider_ids: pd.Series,
    output_dir: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    """Fit models on training providers, select on validation providers, report once on holdout test."""
    y = pd.Series(labels, index=features.index).astype(int)
    ids = pd.Series(provider_ids, index=features.index).astype(str)
    if len(features) < 100 or y.nunique() != 2 or y.value_counts().min() < 10:
        raise ValueError("Need at least 100 providers and at least 10 examples of each label.")
    columns = list(features.columns)
    X = features.replace([np.inf, -np.inf], np.nan).astype(float)
    # 60% train, 20% validation, 20% test; providers never appear in more than one split.
    X_trainval, X_test, y_trainval, y_test, id_trainval, id_test = train_test_split(
        X, y, ids, test_size=0.20, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_trainval, y_trainval, id_trainval, test_size=0.25, random_state=seed, stratify=y_trainval
    )
    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2500, C=0.5, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=350, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced_subsample", random_state=seed, n_jobs=1,
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=350, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced", random_state=seed, n_jobs=1,
            )),
        ]),
    }
    candidates: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = float(np.quantile(val_prob[y_val.to_numpy() == 0], 0.95))
        candidates[name] = {
            "validation_average_precision": float(average_precision_score(y_val, val_prob)),
            "validation_roc_auc": float(roc_auc_score(y_val, val_prob)),
            "threshold_calibrated_to_5pct_validation_fpr": threshold,
        }
        fitted_models[name] = model
    selected_name = max(candidates, key=lambda name: (candidates[name]["validation_average_precision"], candidates[name]["validation_roc_auc"]))
    selected_model = fitted_models[selected_name]
    selected_threshold = candidates[selected_name]["threshold_calibrated_to_5pct_validation_fpr"]

    test_results: dict[str, Any] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for name, model in fitted_models.items():
        prob = model.predict_proba(X_test)[:, 1]
        test_probabilities[name] = prob
        test_results[name] = _metrics(y_test.to_numpy(), prob, candidates[name]["threshold_calibrated_to_5pct_validation_fpr"])
    selected_prob = test_probabilities[selected_name]

    # Unsupervised comparator, threshold calibrated to a 5% validation false-positive rate.
    iso_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(n_estimators=350, contamination="auto", random_state=seed, n_jobs=1)),
    ])
    iso_pipeline.fit(X_train)
    val_anomaly = -iso_pipeline.decision_function(X_val)
    iso_threshold = float(np.quantile(val_anomaly[y_val.to_numpy() == 0], 0.95))
    test_anomaly = -iso_pipeline.decision_function(X_test)
    isolation_metrics = _metrics(y_test.to_numpy(), test_anomaly, iso_threshold)

    # Feature importance is explanatory context, not proof of causal fraud indicators.
    estimator = selected_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_, dtype=float)).mean(axis=0)
    else:
        importance = np.zeros(len(columns), dtype=float)
    top_features = [
        {"feature": str(columns[i]), "importance": float(importance[i])}
        for i in np.argsort(importance)[::-1][:15]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame({
        "Provider": id_test.to_numpy(),
        "PotentialFraud_true": y_test.to_numpy(),
        "selected_model": selected_name,
        "supervised_risk_score": selected_prob,
        "supervised_flag_at_5pct_validation_fpr": selected_prob >= selected_threshold,
        "isolation_forest_anomaly_score": test_anomaly,
        "isolation_forest_flag_at_5pct_validation_fpr": test_anomaly >= iso_threshold,
    }).sort_values("supervised_risk_score", ascending=False)
    prediction_path = output_dir / "public_provider_test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    model_dir = output_dir.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / "public_provider_fraud_model.joblib"
    joblib.dump({
        "model": selected_model,
        "model_name": selected_name,
        "threshold": selected_threshold,
        "feature_columns": columns,
        "label": "PotentialFraud (provider-level, as supplied by dataset)",
        "seed": seed,
        "dataset_source": DATA_SOURCE,
    }, bundle_path, compress=3)

    report = {
        "experiment": "public Medicare healthcare provider fraud-risk classification",
        "dataset_source": DATA_SOURCE,
        "mirror_source": REPOSITORY,
        "dataset_citation_and_license_reference": DATASET_MENDELEY,
        "label_grain": "provider-level PotentialFraud Yes/No; not claim-level fraud-mechanism ground truth",
        "interpretation_warning": "A positive label is the dataset's PotentialFraud designation, not a verified per-claim adjudication and not a label for the four OW-ERF synthetic mechanisms.",
        "seed": seed,
        "python_version": __import__("platform").python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "provider_count": int(len(y)),
        "label_counts": {"not_flagged": int((y == 0).sum()), "potential_fraud": int((y == 1).sum())},
        "positive_prevalence": float(y.mean()),
        "feature_count": int(len(columns)),
        "split": {
            "train_providers": int(len(X_train)),
            "train_positive": int(y_train.sum()),
            "validation_providers": int(len(X_val)),
            "validation_positive": int(y_val.sum()),
            "test_providers": int(len(X_test)),
            "test_positive": int(y_test.sum()),
            "test_negative": int((y_test == 0).sum()),
            "random_seed": seed,
            "split_rule": "stratified provider-level 60/20/20 train/validation/test; no provider crosses splits",
        },
        "models_selected_on_validation": candidates,
        "selected_model": selected_name,
        "selected_model_test_metrics_at_validation_5pct_fpr_threshold": test_results[selected_name],
        "all_supervised_models_test_metrics": test_results,
        "isolation_forest_test_metrics_at_validation_5pct_fpr_threshold": isolation_metrics,
        "top_features": top_features,
        "dataset_file_sha256": {},
        "artifacts": {
            "predictions": str(prediction_path.as_posix()),
            "model": str(bundle_path.as_posix()),
        },
        "limitations": [
            "The target is at provider level; results do not validate claim-level open-world detection or the four individual synthetic fraud mechanisms.",
            "PotentialFraud is a benchmark designation, not proof that every claim associated with a provider is fraudulent.",
            "The GitHub mirror is used to download the labelled-train tables at runtime; the raw tables are not committed here.",
            "Performance is for a single seed and a held-out provider split; repeated seeds, temporal splits, calibration study, and independent external replication are future work.",
        ],
    }
    report["dataset_file_sha256"] = getattr(train_test_provider_model, "_dataset_hashes", {})
    report_path = output_dir / "public_provider_fraud_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def run_dataset_experiment(
    data_dir: Path,
    output_dir: Path,
    seed: int = SEED,
    force_download: bool = False,
) -> dict[str, Any]:
    hashes = ensure_dataset(data_dir, force_download=force_download)
    labels = pd.read_csv(data_dir / FILES["labels"], low_memory=False)
    beneficiaries = pd.read_csv(data_dir / FILES["beneficiary"], low_memory=False)
    inpatient = _read_claims(data_dir / FILES["inpatient"])
    outpatient = _read_claims(data_dir / FILES["outpatient"])
    features, y, ids = engineer_provider_features(inpatient, outpatient, beneficiaries, labels)
    train_test_provider_model._dataset_hashes = hashes
    report = train_test_provider_model(features, y, ids, output_dir, seed=seed)
    report["dataset_file_sha256"] = hashes
    (output_dir / "public_provider_fraud_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _toy_frame(seed: int = 42, n: int = 300) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Deterministic small fixture used by tests; not used as project results."""
    rng = np.random.default_rng(seed)
    labels = pd.Series(np.array([0] * int(n * 0.9) + [1] * (n - int(n * 0.9))))
    ids = pd.Series([f"TOY{i:05d}" for i in range(n)])
    x = pd.DataFrame({
        f"feature_{j}": rng.normal(loc=labels.to_numpy() * (0.15 + j / 10), size=n)
        for j in range(12)
    })
    x["feature_signal"] = labels.to_numpy() + rng.normal(0, 0.15, n)
    x.index = pd.Index(ids, name="Provider")
    labels.index = x.index
    ids.index = x.index
    return x, labels, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/external/raw", help="Directory for downloaded labelled-train CSVs (not committed).")
    parser.add_argument("--output-dir", default="results", help="Directory for reports, predictions and model bundle.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    report = run_dataset_experiment(Path(args.data_dir), Path(args.output_dir), seed=args.seed, force_download=args.force_download)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
