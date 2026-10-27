"""Interactive Insurance Intelligence prototype.

Run with: streamlit run app.py
The policy catalogue shipped with this repo is fictional; the risk models are research prototypes.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from policy_recommender import recommend_policies

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Insurance Intelligence", page_icon="🛡️", layout="wide")
st.title("Insurance Intelligence")
st.caption("Personalized policy matching + open-world fraud-risk research prototype")
st.warning(
    "Research/demo software only. Bundled policy products are fictional. "
    "Fraud scores are triage hypotheses, not findings of wrongdoing or grounds to deny a claim."
)

def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


catalog_default = read_json(ROOT / "data/demo_policy_catalog.json", [])
profile_default = read_json(ROOT / "data/demo_customer_profile.json", {})
tab_policy, tab_owerf, tab_public = st.tabs([
    "Policy recommendation", "OW-ERF open-world pilot", "Public provider benchmark"
])

with tab_policy:
    st.subheader("Find policies that fit your needs")
    st.caption(
        "Compatibility scores are weighted rule-based matches, not purchase advice or approval probabilities. "
        "Upload a verified structured policy catalogue if you have one; clause parsing remains a heuristic."
    )
    upload = st.file_uploader("Optional structured policy catalogue (JSON)", type=["json"], key="policy_catalogue")
    catalog = catalog_default
    if upload is not None:
        try:
            uploaded = json.loads(upload.getvalue().decode("utf-8"))
            if not isinstance(uploaded, list) or not uploaded:
                raise ValueError("Catalogue must be a non-empty JSON list.")
            catalog = uploaded
            st.success(f"Loaded {len(catalog)} policies from the uploaded JSON.")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            st.error(f"Could not load catalogue: {exc}")
    with st.form("policy_profile"):
        left, right = st.columns(2)
        with left:
            age = st.slider("Age of primary member", 18, 80, int(profile_default.get("age", 34)))
            family_size = st.slider("Family members to cover", 1, 10, int(profile_default.get("family_size", 4)))
            budget = st.number_input("Annual premium budget (₹)", min_value=1000, max_value=1000000,
                                     value=int(profile_default.get("annual_budget_inr", 25000)), step=1000)
            sum_insured = st.selectbox("Desired sum insured", [300000, 500000, 1000000, 1500000, 2000000, 2500000, 5000000],
                                       index=2)
            city = st.text_input("City for cashless-network comparison", profile_default.get("city", "Jaipur"))
        with right:
            health_needs = st.multiselect(
                "Known health needs / pre-existing conditions",
                ["diabetes", "hypertension", "cardiac condition", "asthma", "thyroid condition"],
                default=[x for x in profile_default.get("health_needs", []) if x in
                         ["diabetes", "hypertension", "cardiac condition", "asthma", "thyroid condition"]],
            )
            benefits = st.multiselect(
                "Required benefits",
                ["maternity", "daycare", "no_room_rent_cap", "cashless_hospital", "home_care", "restoration_benefit"],
                default=[x for x in profile_default.get("required_benefits", []) if x in
                         ["maternity", "daycare", "no_room_rent_cap", "cashless_hospital", "home_care", "restoration_benefit"]],
            )
            max_wait = st.slider("Target maximum pre-existing-condition wait (months)", 0, 48,
                                 int(profile_default.get("max_pre_existing_wait_months", 24)), step=6)
            no_room_cap = st.checkbox("Prefer no room-rent cap", bool(profile_default.get("prefer_no_room_rent_cap", True)))
            min_hospitals = st.number_input("Minimum cashless hospitals desired", min_value=1, max_value=2000,
                                            value=int(profile_default.get("min_cashless_hospitals", 50)), step=10)
            top_k = st.slider("Policies to display", 1, min(10, len(catalog)) if catalog else 1, min(3, len(catalog)) if catalog else 1)
        submitted = st.form_submit_button("Rank matching policies", type="primary")
    if submitted:
        profile = {
            "age": age, "family_size": family_size, "annual_budget_inr": budget,
            "desired_sum_insured_inr": sum_insured, "city": city,
            "health_needs": health_needs, "required_benefits": benefits,
            "max_pre_existing_wait_months": max_wait,
            "prefer_no_room_rent_cap": no_room_cap,
            "min_cashless_hospitals": min_hospitals,
        }
        try:
            result = recommend_policies(profile, catalog, top_k=top_k)
            for rank, policy in enumerate(result["recommendations"], start=1):
                with st.container(border=True):
                    head1, head2 = st.columns([3, 1])
                    head1.markdown(f"### {rank}. {policy['policy_name']}")
                    head1.caption(policy.get("issuer", "Issuer not specified"))
                    head2.metric("Compatibility", f"{policy['compatibility_score']:.1f}/100")
                    if not policy["eligible_by_catalogue_rules"]:
                        st.error("Outside the catalogue eligibility limits")
                    st.write(
                        f"**Illustrative annual premium:** ₹{policy['annual_premium_inr']:,.0f} · "
                        f"**Sum insured:** ₹{policy['sum_insured_inr']:,.0f}"
                    )
                    score_cols = st.columns(3)
                    for col, key in zip(score_cols, ["affordability", "coverage", "benefits"]):
                        col.metric(key.replace("_", " ").title(), f"{policy['category_scores'][key]:.0f}/100")
                    good, caution = st.columns(2)
                    with good:
                        st.markdown("**Why it matched**")
                        if policy["reasons"]:
                            for reason in policy["reasons"]:
                                st.write(f"✓ {reason}")
                        else:
                            st.write("No positive reason recorded by this catalogue.")
                    with caution:
                        st.markdown("**Check before buying**")
                        if policy["cautions"]:
                            for item in policy["cautions"]:
                                st.write(f"⚠️ {item}")
                        else:
                            st.write("No caution was triggered by the provided fields; verify full policy wording.")
                    if policy.get("source_url"):
                        st.caption(f"Catalogue source: {policy['source_url']}")
            st.info(result["disclaimer"])
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Profile/catalogue validation error: {exc}")
    elif not catalog:
        st.info("Add a policy catalogue JSON to run recommendations.")

with tab_owerf:
    st.subheader("OW-ERF v0.2 temporal-relational pilot")
    st.write(
        "Runs a controlled synthetic fold: temporal train/selection/calibration/test windows, "
        "episodic pseudo-novel training, energy + Mahalanobis novelty, conformal rejection, "
        "legitimate-only explained-rarity filtering, and clustering."
    )
    st.caption(
        "This pilot uses parametric synthetic data—not Synthea output or verified HBP 2.2 rates—and "
        "is not a full PyG HGT/R-GCN."
    )
    report_path = ROOT / "results/owerf_v02_pilot/owerf_v02_synthetic_benchmark.json"
    saved = read_json(report_path, {})
    if saved.get("folds"):
        st.markdown("**Saved four-mechanism report**")
        st.dataframe([{
            "held_out": ", ".join(fold["hidden_mechanisms"]),
            "ROC-AUC": fold["OW_ERF_novelty_metrics"]["roc_auc"],
            "Average precision": fold["OW_ERF_novelty_metrics"]["average_precision"],
            "novel recall after filter": fold["novel_recall_at_alpha"],
            "clinical false alarm": fold["clinical_false_alarm_rate_after_filter"],
            "escalated": fold["grouping"]["escalated_claims"],
        } for fold in saved["folds"]], use_container_width=True)
        with st.expander("Saved report JSON"):
            st.json(saved)
    else:
        st.info("No saved v0.2 aggregate report is present in this checkout yet.")
    with st.form("owerf_smoke"):
        hidden = st.selectbox("Mechanism held out until deployment", ["collusion", "upcoding", "unbundling", "phantom_billing"])
        n_samples = st.selectbox("Generated claims", [1200, 1600, 2400, 4000], index=1)
        epochs = st.selectbox("Training epochs", [1, 2, 3, 5], index=1)
        tier = st.selectbox("Camouflage tier", [1, 2, 3], index=1)
        run = st.form_submit_button("Run one synthetic OW-ERF fold")
    if run:
        with st.spinner("Training and evaluating on synthetic data…"):
            try:
                from owerf_v02_benchmark import run_fold
                result = run_fold(n_samples=n_samples, epochs=epochs, hidden_mechanisms=(hidden,),
                                  tier=tier, seed=42, output_dir=None)
                st.metric("Novelty ROC-AUC", f"{result['OW_ERF_novelty_metrics']['roc_auc']:.3f}")
                st.metric("Novel recall after filter", f"{result['novel_recall_at_alpha']:.3f}")
                st.json(result)
            except ImportError:
                st.error("Install CPU PyTorch to run this tab: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
            except Exception as exc:
                st.error(f"Experiment failed: {type(exc).__name__}: {exc}")

with tab_public:
    st.subheader("Public provider-level benchmark")
    st.caption("This separately predicts provider-level PotentialFraud labels. It is not claim-level OW-ERF mechanism validation.")
    public_path = ROOT / "results/public_provider_fraud_report.json"
    public = read_json(public_path, {})
    if public:
        model = public.get("selected_model", "not recorded")
        metrics = public.get("selected_model_test_metrics_at_validation_5pct_fpr_threshold", {})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Selected model", model)
        k2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        k3.metric("Average precision", f"{metrics.get('average_precision', 0):.3f}")
        k4.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        st.json(public)
    else:
        st.info("Run scripts/train_public_provider_dataset.py to generate the public benchmark report.")
