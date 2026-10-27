#!/usr/bin/env python3
"""Validate a manually transcribed, versioned HBP package-rate catalogue."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REQUIRED = (
    "package_code", "package_name", "specialty", "rate_tier", "rate_inr",
    "source_document", "source_page", "verified_by", "verified_on"
)


def validate(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = [str(x).strip() for x in (reader.fieldnames or [])]
        missing = [name for name in REQUIRED if name not in fields]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        rows = list(reader)
    if not rows:
        raise ValueError("Catalogue has no data rows. Do not report placeholder rates as official HBP rates.")
    seen: set[tuple[str, str]] = set()
    errors: list[str] = []
    for line_no, row in enumerate(rows, start=2):
        code = (row.get("package_code") or "").strip()
        tier = (row.get("rate_tier") or "").strip()
        name = (row.get("package_name") or "").strip()
        source = (row.get("source_document") or "").strip()
        reviewer = (row.get("verified_by") or "").strip()
        verified_on = (row.get("verified_on") or "").strip()
        if not code or not name or not tier:
            errors.append(f"row {line_no}: package_code, package_name and rate_tier are required")
        key = (code, tier)
        if key in seen:
            errors.append(f"row {line_no}: duplicate package_code/rate_tier {key}")
        seen.add(key)
        try:
            rate = float(row.get("rate_inr", ""))
            if rate <= 0:
                errors.append(f"row {line_no}: rate_inr must be positive")
        except (TypeError, ValueError):
            errors.append(f"row {line_no}: rate_inr must be numeric")
        try:
            page = int(row.get("source_page", ""))
            if page < 1:
                errors.append(f"row {line_no}: source_page must be >= 1")
        except (TypeError, ValueError):
            errors.append(f"row {line_no}: source_page must be an integer")
        if not source or not reviewer or not verified_on:
            errors.append(f"row {line_no}: source_document, verified_by and verified_on are mandatory provenance")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "status": "validated_structure_and_provenance",
        "rows": len(rows),
        "unique_package_code_rate_tiers": len(seen),
        "warning": "This script checks structure and provenance fields, not whether human-transcribed rates match the official PDF. A reviewer must verify each source page.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Reviewed CSV catalogue")
    parser.add_argument("--output", help="Optional validation report JSON")
    args = parser.parse_args()
    report = validate(Path(args.input))
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
