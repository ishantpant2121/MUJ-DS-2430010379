# Insurance Intelligence — Product Requirements Document

**Owner:** Ishant Pant  
**Research supervisor:** Dr. Sandeep Gupta  
**Product type:** Individual B.Tech Data Science research / software prototype

## 1. Product vision

Insurance Intelligence brings two related insurance workflows into one application:

1. **Customer side — personalized policy recommendation:** help a person compare structured insurance-policy information against their age, family, budget, coverage requirement, health needs and preferences. The application should show why each policy matched and what the user must verify.
2. **Insurer side — open-world claim-fraud research:** triage unfamiliar claims without equating rarity with fraud. OW-ERF asks whether a claim is both unfamiliar and unexplained by clinical and provider evidence, then groups escalated claims into candidate patterns for an investigator.

The shared principle is **explainable decision support**, not automated acceptance/denial or a legal conclusion.

## 2. Problem statement

### 2.1 Customer problem
Insurance products differ in premiums, sum insured, exclusions, waiting periods, co-payments, sub-limits, eligibility and cashless-hospital networks. Ranking only by price or coverage is insufficient; a useful recommendation must reflect the user's stated needs and explain both matches and trade-offs.

### 2.2 Provider problem
Conventional closed-world fraud classifiers force a new fraud mechanism into a known class. Generic anomaly detectors over-flag severe but legitimate cases such as polytrauma, serious infection and chronic dialysis. An audit team also needs groups and interpretable evidence, rather than an unexplained ranked list.

## 3. Goals and measurable success criteria

### Customer recommender
- Accept a user profile and a structured policy catalogue.
- Produce a stable, reproducible ranking with a visible weighted score breakdown.
- Explain fit, unmet requested benefits, waiting-period differences, eligibility conflicts, caps/co-payments and missing network evidence.
- Keep extracted or supplied facts traceable to a source and identify facts requiring manual confirmation.
- Never describe a synthetic demo plan as a real product, quote, or purchase recommendation.

### OW-ERF research module
- Train on benign claims and known mechanisms while keeping a selected mechanism out of train, selection and calibration splits.
- Apply energy/Mahalanobis novelty scoring and known-only conformal calibration.
- Escalate only rejected claims whose clinical or structural rarity exceeds thresholds estimated from legitimate references.
- Group the suspicious pool and return audit-oriented cluster cards.
- Report detection quality, legitimate false alarms, rare-severe false rejection and workload/fairness proxies.

### Deep-learning evaluation
- Implement and independently train LSTM, 1D-CNN and Transformer baselines on the same generated data seeds and held-out-mechanism folds.
- Report known-class accuracy, macro-F1 and balanced accuracy separately from novel-mechanism AUROC, average precision, FPR@95% novel recall and conformal rejection metrics.
- Save per-run reports and selected model checkpoints for reproduction.

## 4. Users and primary journeys

### Journey A — customer policy comparison
1. User enters age, family size, annual budget, desired sum insured, health needs, required benefits, waiting-period preference, location and cashless-network requirement.
2. A policy catalogue supplies eligibility, premiums/illustrative prices, coverage, benefits, waiting periods, exclusions, co-payment/room-rent restrictions, network evidence and source URL.
3. The system calculates compatibility categories, ranks eligible plans and explains positive matches and cautions.
4. User opens the source policy wording and verifies all terms before acting.

**Current implementation:** policy_recommender.py, data/demo_customer_profile.json, data/demo_policy_catalog.json, and the Policy Recommendation tab in app.py. The demo plans are fictional. Clause extraction is a rule/keyword baseline; the application does not issue a live premium quote and is not yet a validated semantic legal-document parser.

### Journey B — claim-fraud research / investigator
1. A synthetic claim event is associated with past provider, hospital, doctor, beneficiary, package and referral context.
2. LSTM/CNN/Transformer representations or a relational pilot produce a known-class representation.
3. Energy and Mahalanobis novelty are fused and calibrated using known-only data.
4. Claims that pass the conformal novelty gate are checked against legitimate-only clinical and structural references.
5. The suspicious pool is grouped; a cluster card shows timing, provider concentration, risk/reason metrics and deviant observable features.
6. An investigator reviews the evidence. The software must not automatically deny a claim or accuse a provider of fraud.

**Current implementation:** owerf_deep_experiment.py (fused sequence-model prototype), scripts/train_evaluate_deep_models.py (independent LSTM/CNN/Transformer benchmark), and owerf_v02_benchmark.py (temporal relation-summary pilot). The complete arbitrary-node PyG HGT/R-GCN system described in the manuscript is a separate extension; the current relation-summary pilot is not represented as a full HGT/R-GCN.

## 5. Functional requirements

| ID | Requirement | Acceptance check |
|---|---|---|
| FR-01 | Validate required profile fields and catalogue structure | Invalid inputs raise readable validation errors |
| FR-02 | Rank policies by a transparent weighted compatibility score | Output includes component scores, weights and ranking |
| FR-03 | Return matched benefits, missing/unconfirmed benefits, reasons and cautions | Tests cover positive and negative fit cases |
| FR-04 | Preserve source links and identify heuristic extraction facts | Output contains source and manual-verification metadata |
| FR-05 | Generate synthetic OW-ERF cases with hidden labels kept out of model features | Tests check holdout and label-separation invariants |
| FR-06 | Train and test standalone LSTM, CNN and Transformer | All three have trained weights and per-run test metrics |
| FR-07 | Run four leave-one-mechanism-out folds over three generator seeds | CI validates 36 model/fold/seed fits |
| FR-08 | Emit JSON and CSV comparison reports | CI checks files, metric ranges and run count |
| FR-09 | Run the OW-ERF v0.2 graph-style pilot | CI checks four folds and generated result report |
| FR-10 | Provide an interactive demo | streamlit run app.py launches policy, OW-ERF, public provider, and deep-model tabs |
| FR-11 | Restrict outputs to decision support | UI and reports state that synthetic scores are not legal findings and do not trigger claim denial |

## 6. Non-functional requirements

- **Reproducibility:** record seeds, sample counts, split sizes, metric definitions, training settings and artifact paths.
- **Leakage control:** no hidden mechanism labels in model inputs; use non-overlapping train/selection/calibration/test windows where specified; use past-only relation summaries.
- **Explainability:** report score components, source facts, reasons and cautions.
- **Privacy:** bundled experiments operate on synthetic data; never commit personal health/claim records.
- **Safety and fairness:** outputs are triage hypotheses for human review. Monitor false alerts by clinically rare cohort and new-vs-established provider proxy.
- **Maintainability:** tests and GitHub Actions validate changes; generated artifacts must be reproducible from scripts.
- **Evidence integrity:** synthetic benchmark metrics, provider-level public-data metrics and any future real claim-level evaluation must be reported separately.

## 7. Project data and ML design

### Customer side
Structured user profile + policy catalogue → validation → weighted compatibility scoring → ranked recommendations with explanations. The current score is a compatibility heuristic, not a probability of suitability.

### Claim side
Synthetic event generator → timestamped claim/entity relations → model representations → novelty score → conformal gate → explained-rarity filter → escalated pool → density grouping → cluster cards and audit metrics.

### Three key architectures
- **LSTM:** sequences of recent provider/beneficiary claim-history features; intended to model temporal rhythm, bursts and drift.
- **1D-CNN:** ordered procedure/line-item tokens; intended to find local billing/code patterns such as unbundling.
- **Transformer:** claim, hospital, doctor, package, diagnosis and neighbour-context tokens; self-attention learns which relationships influence the representation.

The standalone benchmark intentionally trains each architecture independently. The existing fused encoder is a different experiment; results across the two designs must not be conflated.

## 8. Current evidence boundary

The standalone deep benchmark is trained on a parameterised synthetic generator, not real insurance claim records. Its four folds hold out upcoding, unbundling, phantom billing or collusion, each across three generator seeds. The code reports known-class classification and held-out-mechanism novelty as different tasks. Accuracy does not mean fraud-detection accuracy on real claims.

The v0.2 relational pilot adds timestamped relation edges, past-only entity summaries, episodic pseudo-novel energy loss, fused energy/Mahalanobis scoring, split-conformal rejection, a legitimate-only clinical expectation head, structural reference, clustering, cluster cards, a missing-documentation sweep and a provider-history fairness proxy. It still uses parametric synthetic data and is not a full PyG HGT/R-GCN implementation.

The public healthcare-provider dataset benchmark predicts provider-level PotentialFraud; it is not claim-level ground truth for OW-ERF's four mechanisms. It must remain a separate result.

## 9. Ethical and regulatory guardrails

- Do not make automatic claim-denial or settlement decisions.
- Do not treat a novelty score or cluster as proof of fraud.
- Let users inspect the original policy wording and verify exclusions, waiting periods, caps, co-payments, eligibility and network details.
- Clearly label fictional data and uncertainty.
- Do not use a provider-level fraud label as a claim-level fraud label.
- Before real use, perform domain review, robust error/fairness analysis, privacy and security review, and validation on lawful, appropriately labelled data.

## 10. Release and reproducibility commands

Start the UI:

~~~bash
python -m pip install -r requirements-app.txt
streamlit run app.py
~~~

Run the standalone models:

~~~bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/train_evaluate_deep_models.py --n-samples 2400 --epochs 5 --batch-size 128 --seeds 42,43,44 --hidden all --output-dir results/standalone_deep_models
~~~

Run tests:

~~~bash
python -m pytest -v
~~~

Run the v0.2 pilot:

~~~bash
python -m pip install hdbscan
python owerf_v02_benchmark.py --n-samples 2400 --epochs 2 --batch-size 128 --hidden all --tier 2 --seed 42 --output-dir results/owerf_v02_pilot
~~~

## 11. Roadmap

1. Replace fictional catalogue entries with verified, current policy documents and an audited extraction/confirmation flow.
2. Expand the deep benchmark with more seeds, ablations, calibrated threshold sweeps and error analysis.
3. Integrate pinned Synthea trajectories and a versioned condition-to-ICD-10 mapping; curate and verify official AB-PMJAY HBP 2.2 package data rather than approximating rates.
4. Implement and evaluate a full temporal heterogeneous graph encoder (PyG HGT/R-GCN with explicit typed nodes/edges and temporal neighbour sampling).
5. Add the manuscript's complete baseline set and preregistered experiments, including shortcut/leakage audit and group-wise error analysis.
6. Evaluate on a lawful, properly labelled real-world dataset before making generalisation claims.
