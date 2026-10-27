# MUJ-DS-2430010379

## Student details

| Field | Details |
|---|---|
| **Name** | Ishant Pant |
| **Registration Number** | 2430010379 |
| **Branch** | Data Science |
| **Batch** | D |
| **Project Title** | Insurance Intelligence: Personalized Policy Recommendation + Open-World Fraud Detection |
| **GitHub Username** | [ishantpant2121](https://github.com/ishantpant2121) |
| **Training Program** | MUJ-DS Training Program, Batch D |

## About this repository

This repository contains an individual **Insurance Intelligence** prototype led and maintained by Ishant Pant, with Dr. Sandeep Gupta identified as research supervisor. It now contains two separate modules: (1) customer-facing policy recommendation from a policy catalogue and (2) provider/claim fraud-risk research under the OW-ERF open-world methodology.

The bundled policy catalogue and the OW-ERF v0.2 graph pilot use illustrative synthetic data. The public Medicare provider benchmark is a separate provider-level classification task and does not validate the four claim-level OW-ERF mechanisms.

## Repository structure

- `.github/workflows/` — automated Python tests and research-paper PDF build
- `paper/` — LaTeX manuscript, bibliography, and compiled research-paper PDF
- `docs/INSURANCE_INTELLIGENCE_PRD.md` — product vision, goals, user journeys, requirements and guardrails
- `docs/DEEP_LEARNING_MODELS.md` — architecture, training/test protocol and metric definitions for LSTM/CNN/Transformer
- `resources/` — screenshots of experiment/test runs
- `results/` — canonical recorded experiment results
- `tests/` — tests for the policy recommender, tabular/relational OW-ERF and deep-learning benchmarks
- `policy_recommender.py` — explainable policy catalogue ranking with reasons, cautions and rule-based clause extraction
- `data/demo_policy_catalog.json` and `data/demo_customer_profile.json` — fictional sample inputs for the recommender
- `owerf_core.py` — earlier tabular OW-ERF prototype
- `owerf_deep_encoders.py` and `owerf_deep_experiment.py` — fused LSTM, 1D-CNN and Transformer synthetic training pipeline
- `owerf_v02_benchmark.py` — new temporal causal-relational pilot matching the main stages in manuscript v0.2
- `scripts/train_public_provider_dataset.py` — separate public provider-level train/validation/test pipeline
- `requirements.txt` — common data and ML dependencies
- `requirements-deep.txt` — optional dependencies for experimental deep-learning branches
- `INSTALLATION.md` — setup, run, and test instructions
- `app.py` — interactive Streamlit application

Project specification and model guide: [Insurance Intelligence PRD](docs/INSURANCE_INTELLIGENCE_PRD.md) · [Deep-learning architecture and evaluation guide](docs/DEEP_LEARNING_MODELS.md) · [OW-ERF v0.2 implementation map](docs/OW_ERF_V02_IMPLEMENTATION_MAP.md)

No generic empty placeholder folders are tracked. Large reproducible intermediate feature CSVs and the stale text duplicate of the earlier tabular metrics were removed from version control; the training workflow still regenerates feature tables for local use and uploads them with CI artifacts. The project does not require a separate capstone repository.

## Part 1 — Personalized policy recommendation

The user-facing recommender ranks catalogue records by a transparent weighted compatibility score for affordability, requested sum insured, benefits, pre-existing-condition waiting periods, restrictions and local cashless-network count. Each ranked policy includes a score breakdown, fit reasons, missing benefits and cautions.

Run the demo:

```bash
python policy_recommender.py --profile data/demo_customer_profile.json --catalog data/demo_policy_catalog.json --top-k 3 --output results/policy_recommendations_demo.json
python -m pytest tests/test_policy_recommender.py -v
```

**Important:** the four bundled plans are fictional test fixtures, not real insurer products. Clause extraction is a small rule/keyword baseline, not semantic/legal interpretation. For real use, load verified current policy wording and official product/customer-information sheets. IRDAI advises checking room-rent/ICU limits, pre-existing-condition and specific waiting periods, exclusions, co-payments, sub-limits and network hospitals: [IRDAI health-insurance guidance](https://irdai.gov.in/health-dept).

## Launch the interactive application

Install the optional UI dependency and start the web app:

~~~bash
python -m pip install -r requirements-app.txt
streamlit run app.py
~~~

The app has policy recommendation, OW-ERF pilot, public provider benchmark, and standalone LSTM/CNN/Transformer comparison tabs. The model comparison tab displays the saved CI metrics after the training workflow has published its artifacts. The bundled policy catalogue is fictional and should not be used to buy insurance.

## Research paper

[**View or download the compiled research paper PDF**](paper/OWERF_Research_Paper.pdf)

The PDF is built from `paper/main.tex` and `paper/references.bib`. GitHub Actions rebuilds the PDF when the manuscript, bibliography, or build workflow changes.

## Dataset feasibility

[Public dataset review](DATASET_REVIEW.md) records the separate provider-level benchmark that has been run on the Healthcare Provider Fraud Detection Analysis dataset. Its `PotentialFraud` target is provider-level, not per-claim mechanism truth; it is not the evaluation benchmark for the OW-ERF v0.2 synthetic mechanism folds.

## Run the deep prototype

Install the base dependencies and CPU PyTorch, then run a synthetic hidden-mechanism experiment:

```bash
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python owerf_deep_experiment.py --n-samples 4000 --epochs 12 --hidden collusion --seed 42
```

The command trains on three known synthetic fraud mechanisms, calibrates on a separate known-class split, tests on the held-out mechanism, applies outcome/relationship/code-sequence explained-rarity checks, and compares novelty ranking with Isolation Forest. Use `--hidden all` to run the four-mechanism matrix. A single-mechanism run writes `results/deep_prototype_results.json`; the CI matrix is published to that same file on `main`. This is a proof-of-pipeline on generated data, not external validation or a complete temporal graph model.

## Independent LSTM / 1D-CNN / Transformer benchmark

In addition to the fused encoder above, `scripts/train_evaluate_deep_models.py` independently trains and evaluates each architecture using the same synthetic generator and held-out-mechanism protocol. LSTM reads the recent claim-history sequence, 1D-CNN reads ordered procedure-code tokens, and Transformer reads relational-context tokens. For every architecture and fold it reports known-class accuracy, macro-F1, balanced accuracy, held-out mechanism AUROC/Average Precision, FPR@95% novel recall, conformal novel recall, known false-rejection, rare-severe false-rejection, and Precision@K. It saves one checkpoint per model/fold/seed.

Run the full comparison with three generator seeds:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/train_evaluate_deep_models.py --n-samples 2400 --epochs 5 --batch-size 128 --seeds 42,43,44 --hidden all --output-dir results/standalone_deep_models
python -m pytest tests/test_standalone_deep_models.py -v
```

Results: `results/standalone_deep_models/standalone_deep_model_metrics.json` (full reports) and `results/standalone_deep_models/standalone_deep_model_metrics.csv` (one row per model/fold/seed). Trained checkpoints are saved in `results/standalone_deep_models/weights/`. GitHub Actions runs **36 fits**: 3 architectures × 4 withheld mechanisms × 3 generator seeds. The app's **LSTM / CNN / Transformer comparison** tab reads these recorded test metrics.

These are separate model baselines and a synthetic benchmark. The reported known-class accuracy excludes the held-out mechanism; the novelty AUROC/AP treat the withheld fraud mechanism as a novelty-positive class. They are not real-insurance data results, and the three architectures use different input modalities by design.

### Controlled architecture comparison using identical inputs

The native comparison above gives each model its natural input modality, so differences can reflect both architecture and modality. A second, controlled experiment holds input representation fixed: LSTM, a continuous-input 1D-CNN, and Transformer all receive the same standardized relational-context tokens with the same padding mask. Only the architecture/sequence handling differs. Numeric scaling is fitted on training tokens only, padded positions are zeroed, validation loss selects checkpoints, and the test split remains untouched until evaluation.

Reproduce the controlled comparison (25 epochs, 3 seeds, 4 held-out mechanisms for each mode; 72 model fits total):

```bash
python scripts/train_evaluate_deep_models.py --n-samples 2400 --epochs 25 --batch-size 128 --seeds 42,43,44 --hidden all --input-mode native --output-dir results/controlled_deep_model_comparison/native_25ep
python scripts/train_evaluate_deep_models.py --n-samples 2400 --epochs 25 --batch-size 128 --seeds 42,43,44 --hidden all --input-mode shared_context --output-dir results/controlled_deep_model_comparison/shared_context_25ep
python scripts/summarize_deep_comparison.py --native results/controlled_deep_model_comparison/native_25ep/standalone_deep_model_metrics.json --shared-context results/controlled_deep_model_comparison/shared_context_25ep/standalone_deep_model_metrics.json --output results/controlled_deep_model_comparison/controlled_comparison_summary.json
```

The dedicated GitHub Actions workflow validates both 36-run reports, builds a mean/SD comparison, and publishes metrics plus CI checkpoint artifacts. This comparison is synthetic and controls the input tensor; it does not establish real-claim performance or a statistically conclusive ranking with only three seeds.

### Recorded standalone deep-model results

The dedicated CI run trained **36 independent models**: 3 architectures × 4 held-out mechanisms × 3 generator seeds. Values below are means ± standard deviations across the 12 held-out-mechanism/seed runs for each architecture, on synthetic data.

| Model | Known-class accuracy | Known-class macro-F1 | Novel-mechanism AUROC | Novel-mechanism AP | Novel recall at conformal α=0.05 | Known false-rejection rate |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.630 ± 0.183 | 0.380 ± 0.105 | 0.576 ± 0.189 | 0.139 ± 0.277 | 0.156 ± 0.288 | 0.050 ± 0.011 |
| 1D-CNN | 0.565 ± 0.114 | 0.279 ± 0.048 | 0.561 ± 0.126 | 0.092 ± 0.093 | 0.120 ± 0.159 | 0.047 ± 0.011 |
| Transformer | 0.809 ± 0.115 | 0.646 ± 0.137 | 0.693 ± 0.142 | 0.142 ± 0.215 | 0.156 ± 0.251 | 0.045 ± 0.011 |

**Interpretation:** the Transformer is strongest on known-class classification in this initial run, but unseen-mechanism ranking remains only moderate and highly variable across folds. At α=0.05, novel-mechanism recall is low; the gate is calibrated on known classes, not unseen fraud. These results are useful diagnostics, not evidence of real-world fraud performance. Do not report known-class accuracy as a single overall fraud-detection accuracy.

The workflow also records FPR@95% novel recall: LSTM 0.774 ± 0.326, 1D-CNN 0.889 ± 0.154, Transformer 0.670 ± 0.273. Full per-run outputs and trained checkpoints are generated by CI under the paths listed above. See the [successful training workflow](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40473140833).

## Part 2 — OW-ERF v0.2 temporal relational pilot

The new uploaded manuscript is a registered-report-style design document: it is explicitly **pre-implementation** and contains no empirical findings. The implementation is tracked separately from the manuscript so design intent is not confused with results.

Run all four leave-one-mechanism-out folds:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install hdbscan
python owerf_v02_benchmark.py --n-samples 2400 --epochs 2 --batch-size 128 --hidden all --tier 2 --seed 42 --output-dir results/owerf_v02_pilot
python -m pytest tests/test_owerf_v02_benchmark.py -v
```

The pilot implements timestamped claim-to-entity edges, relation-specific attention over **past-only entity summaries**, episodic pseudo-novel energy training, energy/Mahalanobis CDF fusion, split-conformal p-values, a legitimate-only clinical expectation head, legitimate-only structural references, explained-rarity escalation, HDBSCAN clustering (DBSCAN fallback if HDBSCAN is unavailable), and audit-oriented metrics. It produces a report, test predictions, an edge list and a separate hidden-ground-truth table for evaluation.

[Implementation map and gaps](docs/OW_ERF_V02_IMPLEMENTATION_MAP.md) lists each of the eight stages against actual code.

**Scope limitation:** this is a parametric synthetic pilot, not yet the full manuscript benchmark. It does not currently run Synthea or use a curated/validated table of official AB-PMJAY HBP 2.2 package rates; nor does it implement full arbitrary-node PyTorch-Geometric HGT/R-GCN or every baseline, ablation, camouflage sweep and repeated-seed run in the manuscript. Official design references: [Synthea](https://github.com/synthetichealth/synthea) and [NHA HBP 2.2 manual](https://nha.gov.in/img/resources/HBP-2.2-manual.pdf).

## Public data train/test benchmark

A separate supervised **provider-level** benchmark is available in `scripts/train_public_provider_dataset.py`. It downloads the labelled training tables at runtime from the public GitHub mirror of the Kaggle Healthcare Provider Fraud Detection Analysis dataset, builds provider aggregates, trains Logistic Regression / Random Forest / Extra Trees, selects a model using validation average precision, evaluates once on a held-out provider test split, and compares with Isolation Forest. The raw dataset is not committed.

Dataset references:
- [Kaggle Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
- [Public GitHub data mirror](https://github.com/reshmanalage/healthcare-provider-fraud-analytics)
- [Mendeley Data citation / CC BY 4.0 reference](https://data.mendeley.com/datasets/gsn2hyty37/1)

**Important:** the target is provider-level `PotentialFraud` Yes/No, not ground-truth claim labels and not the four OW-ERF synthetic mechanism labels. This is a supplementary external benchmark for provider-risk classification; it does not validate claim-level open-world mechanism discovery.

Run the full public-data training and held-out test experiment:

```bash
python -m pip install -r requirements.txt
python scripts/train_public_provider_dataset.py --seed 42
python scripts/score_provider_features.py
```

By default, raw CSVs are downloaded into ignored `data/external/raw/`. Core retained artifacts are `results/public_provider_fraud_report.json`, `results/public_provider_test_predictions.csv`, and the fitted model at `models/public_provider_fraud_model.joblib`. The reproducible feature tables (`results/public_provider_features.csv` and `results/public_provider_scored_features.csv`) are generated during training/scoring and uploaded as CI artifacts, but are not committed to keep the repository small.

**Latest public benchmark run (seed 42):** 5,410 providers, 506 labelled `PotentialFraud=Yes` (9.35%); 3,246 train / 1,082 validation / 1,082 held-out test providers. The model was selected by validation average precision. Logistic Regression was selected; on the held-out test set it achieved ROC-AUC **0.956**, average precision **0.754**, precision **0.629**, recall **0.723**, F1 **0.673**, and false-positive rate **4.38%** at the threshold calibrated from validation negatives. The Isolation Forest comparison achieved ROC-AUC 0.853 and average precision 0.455 at its corresponding validation-calibrated threshold.

[Open the exact JSON report](results/public_provider_fraud_report.json) · [Open held-out predictions](results/public_provider_test_predictions.csv) · [Open the trained model artifact](models/public_provider_fraud_model.joblib)

These metrics are for **provider-level benchmark classification only**. They do not validate claim-level discovery of upcoding, unbundling, phantom billing, or collusion, and a `PotentialFraud` label is not proof that every claim by a provider is fraudulent.

## Current implementation status

- **Policy recommendation:** working transparent ranking prototype with reasons/cautions and heuristic clause facts; demo policies are fictional.
- **OW-ERF deep model:** trained LSTM/1D-CNN/Transformer synthetic experiment with four held-out mechanisms; see `results/deep_prototype_results.json`.
- **OW-ERF v0.2 method pilot:** temporal relational-context encoder, pseudo-novel energy training, fused novelty, conformal rejection, legitimate-only explanation references, clustering and audit metrics; details in the implementation map and the automated result report.
- **Public-data comparator:** a separate provider-level model has been trained and tested; details in `results/public_provider_fraud_report.json`.
- **Automated CI:** standard tests run on Python 3.11/3.12; dedicated workflows exercise PyTorch, policy ranking and dataset/model pipelines.
- **Still incomplete:** full Synthea ingestion, verified HBP 2.2 package/rate mapping, arbitrary-node PyG HGT/R-GCN, all paper baselines/ablations, camouflage/documentation sweeps and multi-seed confirmatory runs. All synthetic prototype metrics are exploratory and must not be presented as real-claims validation.

## Research authorship

- **Ishant Pant** — lead / first author.
- **Dr. Sandeep Gupta** — research supervisor.

Bibliography entries retain the names of researchers whose publications are cited; those citations credit external research and do not identify additional project team members.
