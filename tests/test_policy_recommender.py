"""Tests for the policy compatibility ranking and clause extraction."""
from __future__ import annotations

import json
from pathlib import Path

from policy_recommender import extract_clause_facts, recommend_policies

ROOT = Path(__file__).resolve().parents[1]


def test_demo_profile_ranks_eligible_policy_and_explains_tradeoffs():
    profile = json.loads((ROOT / "data/demo_customer_profile.json").read_text())
    catalog = json.loads((ROOT / "data/demo_policy_catalog.json").read_text())
    result = recommend_policies(profile, catalog, top_k=4)

    assert len(result["recommendations"]) == 4
    assert result["recommendations"][0]["eligible_by_catalogue_rules"] is True
    assert result["recommendations"][0]["compatibility_score"] > result["recommendations"][-1]["compatibility_score"]
    assert result["recommendations"][0]["requested_benefits_unconfirmed"] == []
    assert result["demo_catalogue"] is True
    assert "fictional" in result["disclaimer"].lower()


def test_clause_rule_extractor_reads_explicit_waiting_period_and_caps():
    facts = extract_clause_facts(
        "Pre-existing disease is covered after 24 months. Room rent capped at 1% of sum insured. "
        "Day-care procedures are covered. Co-payment of 10% applies."
    )
    assert facts["pre_existing_waiting_period_months"] == 24
    assert facts["room_rent_cap_pct"] == 1.0
    assert facts["benefit_daycare"] is True
    assert facts["co_pay_pct"] == 10.0
    assert facts["needs_manual_verification"] is True


def test_missing_requested_benefit_is_reported_as_a_caution():
    profile = {
        "age": 30, "family_size": 2, "annual_budget_inr": 20000,
        "desired_sum_insured_inr": 1000000, "required_benefits": ["maternity"],
        "health_needs": [], "city": "Delhi", "min_cashless_hospitals": 50,
    }
    catalog = [{
        "policy_id": "X", "name": "Example", "demo_only": True,
        "annual_premium_inr": 10000, "sum_insured_inr": 1000000,
        "min_entry_age": 18, "max_entry_age": 65, "max_family_members": 6,
        "benefits": [], "cashless_hospitals_by_city": {},
    }]
    row = recommend_policies(profile, catalog)["recommendations"][0]
    assert row["requested_benefits_unconfirmed"] == ["maternity"]
    assert any("maternity" in caution.lower() for caution in row["cautions"])
