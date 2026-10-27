#!/usr/bin/env python3
"""Explainable health-insurance policy matching from a structured catalogue.

The bundled catalogue is fictional demo data for software testing, not a current
product quote or an authoritative interpretation of an insurance contract.
Clause extraction is a transparent keyword/rule baseline; users must verify
all extracted facts against the insurer's full policy wording.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

WEIGHTS = {
    "affordability": 20.0,
    "coverage": 20.0,
    "benefits": 25.0,
    "waiting_period": 15.0,
    "restrictions": 10.0,
    "cashless_network": 10.0,
}
CLAUSE_EXTRACTOR_VERSION = "rules-v1"


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def extract_clause_facts(text: str) -> dict[str, Any]:
    """Extract a few explicit contract facts with rules; not legal-grade NLP."""
    clause = " ".join(str(text or "").split())
    normal = _normalise(clause)
    facts: dict[str, Any] = {"extractor": CLAUSE_EXTRACTOR_VERSION, "needs_manual_verification": True}
    wait_matches = re.findall(
        r"(?:pre[\s-]?existing (?:disease|condition|illness)[^.]{0,100}?(?:after|of)\s*)(\d+)\s*(months?|years?)",
        clause,
        flags=re.IGNORECASE,
    )
    if wait_matches:
        value, unit = wait_matches[0]
        facts["pre_existing_waiting_period_months"] = int(value) * (12 if unit.lower().startswith("year") else 1)

    if re.search(r"\b(no|without)\s+(?:a\s+)?room[- ]?rent cap\b|\broom[- ]?rent.{0,25}\bno limit\b", normal):
        facts["room_rent_cap_pct"] = None
    else:
        cap = re.search(r"room[- ]?rent.{0,40}?(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:the\s+)?sum insured", clause, flags=re.IGNORECASE)
        if cap:
            facts["room_rent_cap_pct"] = float(cap.group(1))

    if re.search(r"\bday[- ]?care\b", normal) and re.search(r"\bcover(?:ed|age|s)?\b", normal):
        facts["benefit_daycare"] = True
    if re.search(r"\bmaternity\b", normal) and re.search(r"\bcover(?:ed|age|s)?\b", normal):
        facts["benefit_maternity"] = True
    if re.search(r"\bcashless\b", normal) and re.search(r"\bhospital", normal):
        facts["benefit_cashless_hospital"] = True

    copay = re.search(r"co[- ]?payment\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", clause, flags=re.IGNORECASE)
    if copay:
        facts["co_pay_pct"] = float(copay.group(1))

    exclusion_match = re.search(r"exclusions?\s*:\s*(.+?)(?:\.|$)", clause, flags=re.IGNORECASE)
    if exclusion_match:
        facts["exclusions_extracted"] = [x.strip().lower() for x in re.split(r",|;|\band\b", exclusion_match.group(1)) if x.strip()]
    return facts


def _policy_facts(policy: dict[str, Any]) -> dict[str, Any]:
    result = dict(policy)
    clauses = str(result.get("clause_text", ""))
    extracted = extract_clause_facts(clauses) if clauses else {}
    mapping = {
        "pre_existing_waiting_period_months": "pre_existing_waiting_period_months",
        "room_rent_cap_pct": "room_rent_cap_pct",
        "co_pay_pct": "co_pay_pct",
    }
    for source_key, target_key in mapping.items():
        if target_key not in result and source_key in extracted:
            result[target_key] = extracted[source_key]
    benefits = set(map(str.lower, result.get("benefits", [])))
    for benefit, key in (
        ("daycare", "benefit_daycare"),
        ("maternity", "benefit_maternity"),
        ("cashless_hospital", "benefit_cashless_hospital"),
    ):
        if key in extracted and extracted[key]:
            benefits.add(benefit)
    result["benefits"] = sorted(benefits)
    if "exclusions" not in result:
        result["exclusions"] = extracted.get("exclusions_extracted", [])
    result["_clause_extraction"] = extracted
    return result


def _score_policy(profile: dict[str, Any], raw_policy: dict[str, Any]) -> dict[str, Any]:
    policy = _policy_facts(raw_policy)
    name = str(policy.get("name", "Unnamed policy"))
    age = int(profile.get("age", 0))
    family_size = int(profile.get("family_size", 1))
    budget = float(profile.get("annual_budget_inr", 0))
    desired_cover = float(profile.get("desired_sum_insured_inr", 0))
    required_benefits = {_normalise(x).replace(" ", "_") for x in profile.get("required_benefits", [])}
    policy_benefits = {_normalise(x).replace(" ", "_") for x in policy.get("benefits", [])}
    health_needs = {_normalise(x).replace(" ", "_") for x in profile.get("health_needs", [])}
    premium = float(policy.get("annual_premium_inr", 0))
    cover = float(policy.get("sum_insured_inr", 0))

    reasons: list[str] = []
    cautions: list[str] = []
    eligibility_reasons: list[str] = []
    min_age = int(policy.get("min_entry_age", 18))
    max_age = int(policy.get("max_entry_age", 65))
    max_family = int(policy.get("max_family_members", 6))
    eligible = min_age <= age <= max_age and family_size <= max_family
    if age < min_age or age > max_age:
        eligibility_reasons.append(f"Age {age} is outside this demo product's stated entry range ({min_age}–{max_age}).")
    if family_size > max_family:
        eligibility_reasons.append(f"Family size {family_size} exceeds the catalogue limit of {max_family}.")
    if not eligible:
        cautions.extend(eligibility_reasons)

    budget_score = 100.0 if premium <= budget else max(0.0, 100.0 * budget / max(premium, 1.0))
    if premium <= budget:
        reasons.append(f"Illustrative annual premium ₹{premium:,.0f} is within the annual budget ₹{budget:,.0f}.")
    else:
        cautions.append(f"Illustrative annual premium is ₹{premium - budget:,.0f} above the stated budget.")

    cover_score = 100.0 if desired_cover <= 0 else min(100.0, 100.0 * cover / desired_cover)
    if cover >= desired_cover:
        reasons.append(f"Sum insured ₹{cover:,.0f} meets the requested ₹{desired_cover:,.0f}.")
    else:
        cautions.append(f"Sum insured is ₹{desired_cover - cover:,.0f} below the requested amount.")

    covered = sorted(required_benefits & policy_benefits)
    missing = sorted(required_benefits - policy_benefits)
    benefit_score = 100.0 if not required_benefits else 100.0 * len(covered) / len(required_benefits)
    for benefit in covered:
        reasons.append(f"Includes requested benefit: {benefit.replace('_', ' ')}.")
    for benefit in missing:
        cautions.append(f"Requested benefit not confirmed in this catalogue: {benefit.replace('_', ' ')}.")

    waits = policy.get("waiting_period_months_by_condition", {})
    default_wait = int(policy.get("pre_existing_waiting_period_months", 36))
    relevant_waits = {need: int(waits.get(need, default_wait)) for need in health_needs}
    max_wait = int(profile.get("max_pre_existing_wait_months", 24))
    if relevant_waits:
        worst_wait = max(relevant_waits.values())
        waiting_score = max(0.0, min(100.0, 100.0 * max_wait / max(worst_wait, 1)))
        for condition, months in relevant_waits.items():
            if months <= max_wait:
                reasons.append(f"Listed waiting period for {condition.replace('_', ' ')}: {months} months (within the profile's target).")
            else:
                cautions.append(f"Listed waiting period for {condition.replace('_', ' ')}: {months} months (longer than the target {max_wait}).")
    else:
        worst_wait = default_wait
        waiting_score = 100.0
        reasons.append("No pre-existing condition was supplied in the profile; waiting-period fit was not penalised.")

    room_cap = policy.get("room_rent_cap_pct")
    prefer_no_cap = bool(profile.get("prefer_no_room_rent_cap", False))
    restriction_score = 100.0
    if prefer_no_cap and room_cap is not None:
        restriction_score -= 55.0
        cautions.append(f"Room-rent cap of {room_cap}% is listed; room-category limits may affect admissible expenses.")
    elif prefer_no_cap:
        reasons.append("Catalogue indicates no room-rent cap.")
    copay = float(policy.get("co_pay_pct", 0))
    if copay > 0:
        restriction_score -= min(50.0, copay * 2.0)
        cautions.append(f"Co-payment of {copay}% is listed; the member may pay part of an admissible claim.")
    exclusions = [_normalise(x) for x in policy.get("exclusions", [])]
    matched_exclusions = [x for x in exclusions if any(need.replace("_", " ") in x or x in need.replace("_", " ") for need in health_needs)]
    if matched_exclusions:
        restriction_score -= min(60.0, 30.0 * len(matched_exclusions))
        cautions.append("A listed exclusion appears relevant to a stated health need: " + ", ".join(matched_exclusions) + ".")
    restriction_score = max(0.0, restriction_score)

    city = _normalise(profile.get("city", ""))
    network = policy.get("cashless_hospitals_by_city", {})
    city_count = int(network.get(city, network.get(str(profile.get("city", "")), 0)))
    min_network = max(1, int(profile.get("min_cashless_hospitals", 50)))
    network_score = min(100.0, 100.0 * city_count / min_network)
    if city_count:
        reasons.append(f"Catalogue lists {city_count} cashless hospitals for the requested city.")
    else:
        cautions.append("No cashless-hospital count is supplied for the requested city; verify network coverage directly.")

    category_scores = {
        "affordability": budget_score,
        "coverage": cover_score,
        "benefits": benefit_score,
        "waiting_period": waiting_score,
        "restrictions": restriction_score,
        "cashless_network": network_score,
    }
    total = sum(category_scores[key] * WEIGHTS[key] / 100.0 for key in WEIGHTS)
    if not eligible:
        total = 0.0

    return {
        "policy_id": policy.get("policy_id", name),
        "policy_name": name,
        "issuer": policy.get("issuer", "Not specified"),
        "demo_only": bool(policy.get("demo_only", True)),
        "eligible_by_catalogue_rules": eligible,
        "eligibility_reasons": eligibility_reasons,
        "compatibility_score": round(total, 2),
        "score_interpretation": "Transparent weighted compatibility score, not a probability, guarantee, or financial recommendation.",
        "category_scores": {k: round(v, 2) for k, v in category_scores.items()},
        "weights": WEIGHTS,
        "annual_premium_inr": premium,
        "budget_difference_inr": round(budget - premium, 2),
        "sum_insured_inr": cover,
        "requested_benefits_covered": covered,
        "requested_benefits_unconfirmed": missing,
        "listed_waiting_period_months_by_need": relevant_waits,
        "room_rent_cap_pct": room_cap,
        "co_pay_pct": copay,
        "cashless_hospitals_in_city": city_count,
        "reasons": reasons,
        "cautions": cautions,
        "source_url": policy.get("source_url", ""),
        "clause_extraction": policy.get("_clause_extraction", {}),
    }


def recommend_policies(
    profile: dict[str, Any],
    policies: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Rank a catalogue with user-specific reasons and cautions."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not policies:
        raise ValueError("Policy catalogue is empty.")
    required_profile = ("age", "family_size", "annual_budget_inr", "desired_sum_insured_inr")
    missing = [key for key in required_profile if key not in profile]
    if missing:
        raise ValueError(f"Profile missing required field(s): {missing}")
    scored = [_score_policy(profile, policy) for policy in policies]
    scored.sort(key=lambda row: (row["eligible_by_catalogue_rules"], row["compatibility_score"]), reverse=True)
    return {
        "system": "Insurance Intelligence — policy recommendation prototype",
        "recommendation_type": "structured catalogue compatibility ranking",
        "profile_summary": {
            "age": profile["age"],
            "family_size": profile["family_size"],
            "annual_budget_inr": profile["annual_budget_inr"],
            "desired_sum_insured_inr": profile["desired_sum_insured_inr"],
            "health_needs": profile.get("health_needs", []),
            "required_benefits": profile.get("required_benefits", []),
            "city": profile.get("city", ""),
        },
        "catalogue_count": len(policies),
        "demo_catalogue": any(row["demo_only"] for row in scored),
        "top_k": min(top_k, len(scored)),
        "recommendations": scored[:top_k],
        "disclaimer": (
            "Demo catalogue entries are fictional and must not be used to buy insurance. "
            "For real policies, load verified current insurer documents and validate all terms, "
            "eligibility, premiums, exclusions, waiting periods, sub-limits and network hospitals. "
            "Clause extraction is a keyword/rule baseline and is not legal advice."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="data/demo_customer_profile.json")
    parser.add_argument("--catalog", default="data/demo_policy_catalog.json")
    parser.add_argument("--output", default="results/policy_recommendations_demo.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    policies = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    result = recommend_policies(profile, policies, top_k=args.top_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
