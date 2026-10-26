"""Unit tests for the public provider-level benchmark pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_public_provider_dataset import engineer_provider_features, train_test_provider_model


def test_provider_feature_engineering_respects_provider_label_grain():
    labels = pd.DataFrame({
        "Provider": ["PRV1", "PRV2", "PRV3", "PRV4"],
        "PotentialFraud": ["No", "Yes", "No", "Yes"],
    })
    beneficiaries = pd.DataFrame({
        "BeneID": ["B1", "B2", "B3"],
        "DOB": ["1940-01-01", "1935-02-01", "1948-03-01"],
        "Gender": [1, 2, 1],
        "Race": [1, 2, 1],
        "State": [10, 10, 20],
        "County": [101, 101, 202],
        "ChronicCond_Diabetes": [1, 2, 1],
        "RenalDiseaseIndicator": ["Y", "N", "N"],
    })
    inpatient = pd.DataFrame({
        "Provider": ["PRV1", "PRV1", "PRV2", "PRV3"],
        "ClaimID": ["C1", "C2", "C3", "C4"],
        "BeneID": ["B1", "B2", "B2", "B3"],
        "InscClaimAmtReimbursed": [100, 200, 900, 300],
        "DeductibleAmtPaid": [10, 0, 20, 0],
        "AttendingPhysician": ["D1", "D2", "D8", None],
        "ClaimStartDt": ["2009-01-01", "2009-02-01", "2009-03-01", "2009-04-01"],
        "ClaimEndDt": ["2009-01-02", "2009-02-03", "2009-03-04", "2009-04-02"],
        "AdmissionDt": ["2009-01-01", "2009-02-01", "2009-03-01", "2009-04-01"],
        "DischargeDt": ["2009-01-02", "2009-02-03", "2009-03-04", "2009-04-02"],
        "ClmDiagnosisCode_1": ["A", "A", "B", "A"],
        "ClmProcedureCode_1": ["P1", None, "P8", "P1"],
        "DiagnosisGroupCode": ["G1", "G1", "G2", "G1"],
    })
    outpatient = pd.DataFrame({
        "Provider": ["PRV1", "PRV2", "PRV4"],
        "ClaimID": ["O1", "O2", "O3"],
        "BeneID": ["B1", "B2", "B3"],
        "InscClaimAmtReimbursed": [20, 70, 60],
        "DeductibleAmtPaid": [0, 5, 0],
        "AttendingPhysician": ["D1", "D9", "D3"],
        "ClaimStartDt": ["2009-05-01", "2009-06-01", "2009-07-01"],
        "ClaimEndDt": ["2009-05-01", "2009-06-02", "2009-07-01"],
        "ClmDiagnosisCode_1": ["A", "B", "C"],
        "ClmProcedureCode_1": ["P1", "P9", None],
        "DiagnosisGroupCode": ["G1", "G2", "G3"],
    })
    features, y, provider_ids = engineer_provider_features(inpatient, outpatient, beneficiaries, labels)

    assert len(features) == 4
    assert features.index.is_unique
    assert "Provider" not in features.columns
    assert "PotentialFraud" not in features.columns
    assert y.loc["PRV1"] == 0
    assert y.loc["PRV2"] == 1
    assert provider_ids.loc["PRV3"] == "PRV3"
    assert features.loc["PRV1", "ip_claim_count"] == 2
    assert features.loc["PRV1", "op_claim_count"] == 1
    assert features.loc["PRV1", "ip_reimbursement_sum"] == 300
    assert np.isfinite(features.loc["PRV1", "ip_beneficiary_age_mean"])
    assert features.loc["PRV4", "ip_claim_count"] == 0


def test_train_validation_test_training_produces_model_and_holdout_predictions(tmp_path):
    rng = np.random.default_rng(17)
    n = 300
    y = pd.Series(np.array([0] * 270 + [1] * 30), name="target")
    ids = pd.Series([f"PRV{i:05d}" for i in range(n)], name="Provider")
    X = pd.DataFrame({
        f"feature_{j}": rng.normal(loc=y.to_numpy() * (0.2 + j / 15), size=n)
        for j in range(10)
    })
    X["strong_signal"] = y.to_numpy() + rng.normal(0, 0.2, size=n)
    X.index = pd.Index(ids, name="Provider")
    y.index = X.index
    ids.index = X.index

    report = train_test_provider_model(X, y, ids, tmp_path / "results", seed=9)

    assert report["provider_count"] == n
    assert report["split"]["train_providers"] + report["split"]["validation_providers"] + report["split"]["test_providers"] == n
    assert report["selected_model"] in {"logistic_regression", "random_forest", "extra_trees"}
    assert 0 <= report["selected_model_test_metrics_at_validation_5pct_fpr_threshold"]["roc_auc"] <= 1
    assert (tmp_path / "results" / "public_provider_test_predictions.csv").exists()
    assert (tmp_path / "models" / "public_provider_fraud_model.joblib").exists()
    assert (tmp_path / "results" / "public_provider_fraud_report.json").exists()
