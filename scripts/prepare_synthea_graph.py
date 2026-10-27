#!/usr/bin/env python3
"""Convert a Synthea CSV export into typed node/edge tables for graph inspection.

This adapter does not invent fraud labels and does not train a fraud classifier.
It supports Synthea CSV snapshots containing patients/encounters/conditions/
procedures and optional claims/claims_transactions files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{str(k).strip().upper(): (v or "").strip() for k, v in row.items()
                 if k is not None} for row in csv.DictReader(handle)]


def value(row: dict[str, str], *names: str) -> str:
    for name in names:
        got = row.get(name.upper(), "")
        if got:
            return got
    return ""


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> int:
    row_count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            row_count += 1
    return row_count


def node(nodes: list[dict[str, object]], node_type: str, node_id: str,
         event_time: str = "", attributes: dict[str, object] | None = None) -> None:
    if not node_id:
        return
    nodes.append({
        "node_type": node_type, "node_id": node_id, "event_time": event_time,
        "attributes_json": json.dumps(attributes or {}, sort_keys=True, ensure_ascii=False),
    })


def edge(edges: list[dict[str, object]], src_type: str, src_id: str, relation: str,
         dst_type: str, dst_id: str, timestamp: str) -> None:
    if src_id and dst_id:
        edges.append({
            "source_type": src_type, "source_id": src_id, "relation_type": relation,
            "target_type": dst_type, "target_id": dst_id, "timestamp": timestamp,
        })


def prepare(input_dir: Path, output_dir: Path) -> dict[str, object]:
    required = ("patients.csv", "encounters.csv")
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Required Synthea CSV file(s) missing: " + ", ".join(missing))

    patients = read_table(input_dir / "patients.csv")
    encounters = read_table(input_dir / "encounters.csv")
    conditions = read_table(input_dir / "conditions.csv")
    procedures = read_table(input_dir / "procedures.csv")
    claims = read_table(input_dir / "claims.csv")
    transactions = read_table(input_dir / "claims_transactions.csv")

    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    patient_ids: set[str] = set()
    encounter_ids: set[str] = set()
    provider_ids: set[str] = set()
    organization_ids: set[str] = set()
    condition_codes: set[str] = set()
    procedure_codes: set[str] = set()
    claim_ids: set[str] = set()

    for row in patients:
        pid = value(row, "ID", "Id")
        if not pid:
            continue
        patient_ids.add(pid)
        node(nodes, "Beneficiary", pid, attributes={
            "gender": value(row, "GENDER"), "birthdate": value(row, "BIRTHDATE"),
        })

    for row in encounters:
        eid = value(row, "ID", "Id", "ENCOUNTER")
        pid = value(row, "PATIENT")
        when = value(row, "START", "DATE")
        if not eid:
            continue
        encounter_ids.add(eid)
        provider = value(row, "PROVIDER")
        organization = value(row, "ORGANIZATION")
        if provider:
            provider_ids.add(provider)
        if organization:
            organization_ids.add(organization)
        node(nodes, "Admission", eid, when, {
            "encounter_class": value(row, "ENCOUNTERCLASS"),
            "code": value(row, "CODE"), "description": value(row, "DESCRIPTION"),
            "base_encounter_cost": value(row, "BASE_ENCOUNTER_COST"),
            "total_claim_cost": value(row, "TOTAL_CLAIM_COST"),
        })
        edge(edges, "Admission", eid, "FOR_BENEFICIARY", "Beneficiary", pid, when)
        edge(edges, "Admission", eid, "AT_HOSPITAL", "Hospital", organization, when)
        edge(edges, "Admission", eid, "TREATED_BY", "Doctor", provider, when)

    for row in conditions:
        code = value(row, "CODE", "DESCRIPTION")
        pid = value(row, "PATIENT")
        eid = value(row, "ENCOUNTER")
        when = value(row, "START", "DATE")
        cid = value(row, "CODE", "ID")
        if not cid:
            continue
        condition_codes.add(cid)
        node(nodes, "Diagnosis", cid, when, {
            "code": code, "description": value(row, "DESCRIPTION"),
            "system": value(row, "SYSTEM"),
        })
        edge(edges, "Beneficiary", pid, "HAS_DIAGNOSIS", "Diagnosis", cid, when)
        edge(edges, "Admission", eid, "HAS_DIAGNOSIS", "Diagnosis", cid, when)

    for row in procedures:
        code = value(row, "CODE", "DESCRIPTION")
        pid = value(row, "PATIENT")
        eid = value(row, "ENCOUNTER")
        when = value(row, "DATE", "START")
        if not code:
            continue
        procedure_codes.add(code)
        node(nodes, "Package", code, when, {
            "code": code, "description": value(row, "DESCRIPTION"),
            "base_cost": value(row, "BASE_COST"),
        })
        edge(edges, "Admission", eid, "HAS_PROCEDURE", "Package", code, when)
        edge(edges, "Beneficiary", pid, "HAS_PROCEDURE", "Package", code, when)

    claim_patient: dict[str, str] = {}
    for row in claims:
        cid = value(row, "ID", "CLAIMID")
        pid = value(row, "PATIENTID", "PATIENT")
        provider = value(row, "PROVIDERID", "PROVIDER")
        when = value(row, "FROMDATE", "START", "DATE")
        if not cid:
            continue
        claim_ids.add(cid)
        claim_patient[cid] = pid
        node(nodes, "Claim", cid, when, {
            "patient_id": pid, "provider_id": provider,
            "total_claim_cost": value(row, "TOTAL_CLAIM_COST", "TOTALCOST"),
            "source": "Synthea claims.csv",
        })
        edge(edges, "Claim", cid, "FOR_BENEFICIARY", "Beneficiary", pid, when)
        edge(edges, "Claim", cid, "SUBMITTED_BY", "Doctor", provider, when)

    for row in transactions:
        cid = value(row, "CLAIMID", "CLAIM_ID")
        code = value(row, "PROCEDURECODE", "PROCEDURE_CODE", "CODE", "LINEITEMID")
        when = value(row, "FROMDATE", "DATE")
        if code:
            procedure_codes.add(code)
            node(nodes, "Package", code, when, {
                "code": code, "description": value(row, "NOTES"),
                "unit_amount": value(row, "UNITAMOUNT"), "source": "Synthea claims_transactions.csv",
            })
        edge(edges, "Claim", cid, "HAS_LINE_ITEM", "Package", code, when)

    for provider_id in sorted(provider_ids):
        node(nodes, "Doctor", provider_id)
    for org_id in sorted(organization_ids):
        node(nodes, "Hospital", org_id)

    # Entity nodes are unique by (type, ID); repeated observations remain as timestamped edges.
    deduplicated: dict[tuple[str, str], dict[str, object]] = {}
    for row in nodes:
        deduplicated.setdefault((str(row["node_type"]), str(row["node_id"])), row)
    nodes = list(deduplicated.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    node_count = write_csv(
        output_dir / "nodes.csv",
        ["node_type", "node_id", "event_time", "attributes_json"], nodes
    )
    edge_count = write_csv(
        output_dir / "edges.csv",
        ["source_type", "source_id", "relation_type", "target_type", "target_id", "timestamp"], edges
    )
    file_hashes = {}
    for name in ("patients.csv", "encounters.csv", "conditions.csv", "procedures.csv",
                 "claims.csv", "claims_transactions.csv"):
        path = input_dir / name
        if path.exists():
            file_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    type_counts = dict(Counter(str(item["node_type"]) for item in nodes))
    relation_counts = dict(Counter(str(item["relation_type"]) for item in edges))
    manifest = {
        "adapter": "prepare_synthea_graph.py",
        "input_directory": str(input_dir.resolve()),
        "input_sha256": file_hashes,
        "node_rows": node_count,
        "edge_rows": edge_count,
        "node_counts_by_type": type_counts,
        "edge_counts_by_relation": relation_counts,
        "source_row_counts": {
            "patients": len(patients), "encounters": len(encounters), "conditions": len(conditions),
            "procedures": len(procedures), "claims": len(claims),
            "claims_transactions": len(transactions),
        },
        "claim_rows_present": len(claims),
        "fraud_labels_present": False,
        "limitations": [
            "Synthea records are synthetic health-care records and do not contain ground-truth fraud mechanisms.",
            "CSV schemas can vary by exporter/version; inspect the manifest and source data before use.",
            "This adapter writes graph tables for schema/data auditing; it does not train the fraud detector.",
            "Codes/descriptions are preserved as exported and are not asserted to be ICD-10, Indian billing codes, or HBP package identifiers."
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing Synthea CSV exports")
    parser.add_argument("--output-dir", default="data/prepared_synthea_graph")
    args = parser.parse_args()
    result = prepare(Path(args.input_dir), Path(args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
