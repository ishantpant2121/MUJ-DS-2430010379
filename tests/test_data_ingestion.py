from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.prepare_synthea_graph import prepare
from scripts.validate_hbp_catalog import REQUIRED, validate


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_synthea_adapter_writes_typed_graph_and_does_not_invent_fraud_labels(tmp_path):
    source = tmp_path / "synthea"
    output = tmp_path / "graph"
    source.mkdir()
    write_rows(source / "patients.csv", ["Id", "BIRTHDATE", "GENDER"], [
        {"Id": "p1", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
    ])
    write_rows(source / "encounters.csv", [
        "Id", "START", "PATIENT", "ORGANIZATION", "PROVIDER", "ENCOUNTERCLASS", "CODE",
        "DESCRIPTION", "BASE_ENCOUNTER_COST", "TOTAL_CLAIM_COST"
    ], [{
        "Id": "e1", "START": "2022-01-01T10:00:00Z", "PATIENT": "p1",
        "ORGANIZATION": "h1", "PROVIDER": "d1", "ENCOUNTERCLASS": "ambulatory",
        "CODE": "enc-code", "DESCRIPTION": "visit", "BASE_ENCOUNTER_COST": "120",
        "TOTAL_CLAIM_COST": "150"
    }])
    write_rows(source / "conditions.csv",
               ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"], [{
        "START": "2022-01-01", "PATIENT": "p1", "ENCOUNTER": "e1",
        "CODE": "dx1", "DESCRIPTION": "test condition"
    }])
    write_rows(source / "procedures.csv",
               ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "BASE_COST"], [{
        "DATE": "2022-01-01", "PATIENT": "p1", "ENCOUNTER": "e1",
        "CODE": "proc1", "DESCRIPTION": "test procedure", "BASE_COST": "25"
    }])
    write_rows(source / "claims.csv",
               ["Id", "PATIENTID", "PROVIDERID", "FROMDATE", "TOTAL_CLAIM_COST"], [{
        "Id": "c1", "PATIENTID": "p1", "PROVIDERID": "d1",
        "FROMDATE": "2022-01-01", "TOTAL_CLAIM_COST": "150"
    }])
    write_rows(source / "claims_transactions.csv",
               ["CLAIMID", "PROCEDURECODE", "FROMDATE", "UNITAMOUNT"], [{
        "CLAIMID": "c1", "PROCEDURECODE": "proc1",
        "FROMDATE": "2022-01-01", "UNITAMOUNT": "25"
    }])

    manifest = prepare(source, output)
    assert manifest["fraud_labels_present"] is False
    assert manifest["node_counts_by_type"]["Claim"] == 1
    assert manifest["node_counts_by_type"]["Admission"] == 1
    assert manifest["input_sha256"]["patients.csv"]
    edge_rows = list(csv.DictReader((output / "edges.csv").open(encoding="utf-8")))
    assert any(r["relation_type"] == "HAS_DIAGNOSIS" for r in edge_rows)
    assert any(r["relation_type"] == "HAS_LINE_ITEM" for r in edge_rows)
    assert json.loads((output / "manifest.json").read_text())["fraud_labels_present"] is False


def test_hbp_catalog_requires_provenance_and_positive_rates(tmp_path):
    path = tmp_path / "catalog.csv"
    write_rows(path, list(REQUIRED), [{
        "package_code": "PKG-001", "package_name": "Example verified package",
        "specialty": "Example", "rate_tier": "Tier-2", "rate_inr": "15000",
        "source_document": "HBP 2.2 manual", "source_page": "12",
        "verified_by": "reviewer", "verified_on": "2026-10-27"
    }])
    report = validate(path)
    assert report["rows"] == 1
    assert "not whether human-transcribed rates match" in report["warning"]


def test_hbp_catalog_rejects_duplicate_code_tier_and_missing_provenance(tmp_path):
    path = tmp_path / "catalog.csv"
    row = {
        "package_code": "PKG-001", "package_name": "Example package",
        "specialty": "Example", "rate_tier": "Tier-2", "rate_inr": "0",
        "source_document": "", "source_page": "0", "verified_by": "",
        "verified_on": ""
    }
    write_rows(path, list(REQUIRED), [row, row])
    try:
        validate(path)
    except ValueError as exc:
        message = str(exc)
        assert "duplicate package_code/rate_tier" in message
        assert "rate_inr must be positive" in message
        assert "source_document, verified_by and verified_on" in message
    else:
        raise AssertionError("Invalid catalogue should fail validation")
