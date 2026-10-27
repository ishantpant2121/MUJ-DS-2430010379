# Temporal Heterogeneous Graph Research Baseline

## What is implemented

The new owerf_temporal_heterograph.py module adds an explicit typed-node temporal graph encoder using core PyTorch rather than requiring a compiled PyTorch-Geometric extension. The types are Claim, Beneficiary, Hospital, Doctor, Provider, Admission, Package and Diagnosis. Eight typed relation channels connect claims to these entity types, including separate provider and referral-target channels that share the Provider node namespace. The encoder uses relation-specific claim-to-entity projections, time features and recency weighting for historical messages, learned node-type embeddings, log-history-degree features, relation-specific entity-to-query transforms, attention over relation messages, and an open-set classifier plus heteroscedastic clinical-expectation head.

The generated data schema supplies beneficiary, hospital, doctor, package and timestamp identifiers. To complete a stable seven-type graph with the current synthetic generator, Admission is represented by one synthetic admission node per claim and Diagnosis is a severity-bin proxy. These are explicit proxies, not true clinical admissions or ICD-10 diagnoses. Hidden labels, mechanism IDs and ring IDs are never graph features.

## Leakage-control policy

For every query block [start, end), only claim events with index < start may message into entity nodes; entity nodes then send messages into query claims. Claims inside a query block are not allowed to message into entities used by other claims in the same block. This conservative snapshot approach avoids within-block forward leakage. The chronology is train 0–55%, selection 55–65%, calibration 65–80%, and deployment 80–100%; the synthetic generator reserves the held-out mechanism for deployment. Input scalers are fitted only on the training window; the expectation target scaler is fit only on benign training claims; the novelty threshold uses a separate known-only calibration window.

## Run locally

Install the base and deep-model dependencies:

~~~bash
python -m pip install -r requirements.txt -r requirements-deep.txt
python -m pytest -q tests/test_temporal_heterograph.py
python scripts/train_temporal_heterograph.py --n-samples 2400 --epochs 8 --batch-size 64 --patience 3 --seeds 42,43,44 --hidden all --output-dir results/temporal_heterograph_full
~~~

The full command performs 4 held-out-mechanism folds × 3 generator seeds = 12 synthetic runs. It writes per-fold report.json, test_predictions.json, and model_state.pt, plus aggregate JSON/CSV. The Temporal heterogeneous graph benchmark GitHub Actions workflow has a small one-fold smoke run and a manually selectable full run.

## Model metrics

The benchmark reports known-class accuracy and macro-F1 on non-held-out deployment claims, held-out-mechanism AUROC/AP, novel recall at conformal alpha 0.05, known false-rejection rate, rare-severe-benign false rejection, and recall after the explained-rarity gate. The gate is an audit-triage rule and is not a determination of fraud. Report mean and standard deviation across folds; do not conflate known classification with novel-mechanism detection.

## Important limitations

- The experiment is parametric synthetic claim data, not a real claim-level fraud dataset and not Synthea training/evaluation.
- The diagnosis node is currently a severity-bin proxy. The admission node is synthetic and one-to-one with a claim.
- The implementation is a dependency-light R-GCN/HGT-style temporal message-passing encoder; it is not the PyTorch Geometric library's canonical HGTConv/RGCNConv implementation. This is deliberate for reproducible CPU CI. Do not call it a PyG HGT model.
- Synthea data do not supply fraud ground truth. An ingestion/feature check cannot produce claim-level fraud AUROC without a separately labelled evaluation design.
- The NHA HBP 2.2 manual contains package/rate tables, but no rates were automatically guessed or transcribed. Use only a manually checked, versioned catalogue with source page references.
- The completed 12-run benchmark is summarized in docs/TEMPORAL_GRAPH_BENCHMARK_RESULTS.md and results/temporal_heterograph_full/. The full per-fold JSON reports, test predictions and checkpoints are in GitHub Actions run https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40562936336.
- Full baseline/ablation sweeps, Synthea-derived graph checks, official catalogue curation, shortcut audits, group-wise errors, and claim-level real-data validation remain research work.

## Source references

- Synthea project and export formats: https://github.com/synthetichealth/synthea
- Synthea CSV data dictionary: https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary
- National Health Authority HBP 2.2 manual: https://nha.gov.in/img/resources/HBP-2.2-manual.pdf
