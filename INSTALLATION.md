# OW-ERF Installation and Run Guide

This guide covers the policy recommendation demo, the earlier tabular OW-ERF baseline, the deep synthetic experiment, the new OW-ERF v0.2 relational pilot, and the separate public provider-level benchmark.

## Requirements

- Python 3.10 or newer (the reported owner environment used Python 3.13.5)
- pip
- Windows PowerShell, macOS, or Linux

## 1. Get the repository

Clone the repository and move into its directory:

```powershell
git clone https://github.com/ishantpant2121/MUJ-DS-2430010379.git
cd MUJ-DS-2430010379
```

If you already have a local clone, change into that directory and run `git pull` only after checking that you have no uncommitted changes.

## 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, for this terminal session only run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Run the experiment

From the repository root:

```bash
python owerf_core.py
```

The script runs the synthetic leave-one-mechanism-out experiment over multiple seeds and prints a summary table. It may take several minutes. The results depend on the synthetic data generator and are not evidence of performance on real insurance claims.

A saved result table is available at [results/results.txt](results/results.txt). Re-running the experiment may produce slightly different values if code or dependency versions differ.

## 5. Run tests

```bash
python -m pytest -v
```

The CI suite runs the ordinary unit tests on Python 3.11 and 3.12. PyTorch-focused tests run in dedicated workflows that install CPU PyTorch; check the linked Actions run for the exact commit and results.

## 6. Run the deep prototype

The repository now includes a runnable synthetic-data pipeline in `owerf_deep_experiment.py`. It trains the LSTM, 1D-CNN and Transformer branches on generated history, code-sequence and context-token inputs; calibrates a novelty score on a separate known-class split; applies the explained-rarity filter; and compares the ranking with Isolation Forest.

Install the base dependencies and CPU PyTorch:

```bash
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run one held-out-mechanism experiment:

```bash
python owerf_deep_experiment.py --n-samples 4000 --epochs 12 --hidden collusion --seed 42
```

The script writes its measurements to `results/deep_prototype_results.json`. Change `--hidden` to `upcoding`, `unbundling`, or `phantom_billing` to hold out a different mechanism, or use `--hidden all` for a matrix over all four held-out mechanisms. The explained-rarity gate evaluates outcome residuals plus unusual relationship-context and procedure-code profiles. Run the associated tests with:

```bash
python -m pytest tests/test_deep.py tests/test_deep_experiment.py -v
```

This is an experimental synthetic proof-of-pipeline. The input sequences and context are generated proxies, not real claims; the prototype is not the full temporal heterogeneous graph model, and the smoke run is not a publication-grade ablation study. Do not present it as real-world validation.

## 7. Train/test on the public provider-level dataset

Install the base requirements, including pandas and joblib:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python scripts/train_public_provider_dataset.py --seed 42
python scripts/score_provider_features.py
```

The training script downloads four labelled-train CSVs into `data/external/raw/` from the [public GitHub mirror](https://github.com/reshmanalage/healthcare-provider-fraud-analytics) of the [Kaggle dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis). The files are ignored by Git. It builds provider-level aggregates from inpatient/outpatient claims and beneficiary demographics/condition flags, uses stratified provider-level 60/20/20 train/validation/test splits, chooses among Logistic Regression, Random Forest and Extra Trees on validation average precision, calibrates an alert threshold to a 5% validation false-positive rate, and reports test metrics. Isolation Forest is included as an unsupervised comparator.

The workflow writes the report to `results/public_provider_fraud_report.json`, test predictions to `results/public_provider_test_predictions.csv`, provider features to `results/public_provider_features.csv`, scored rows to `results/public_provider_scored_features.csv`, and a reusable model bundle to `models/public_provider_fraud_model.joblib`.

**Label limitation:** `PotentialFraud` is a provider-level benchmark label. It is not verified per-claim fraud ground truth and does not identify upcoding, unbundling, phantom billing, or collusion individually. Treat this as a separate external provider-classification experiment, not as proof that claim-level OW-ERF generalizes.

Latest recorded held-out test run (seed 42): Logistic Regression selected using validation average precision achieved ROC-AUC 0.956, average precision 0.754, precision 0.629, recall 0.723 and F1 0.673, at a validation-calibrated false-positive rate of 4.38%. The exact metrics, split sizes, file hashes and feature names are in `results/public_provider_fraud_report.json`. These are provider-level `PotentialFraud` benchmark results, not claim-level OW-ERF mechanism validation.

## 8. Run the policy recommendation demo

The demo uses a fictional catalogue in `data/demo_policy_catalog.json` and a sample profile in `data/demo_customer_profile.json`. It is a software fixture, not a real product comparison.

```bash
python policy_recommender.py --profile data/demo_customer_profile.json --catalog data/demo_policy_catalog.json --top-k 3 --output results/policy_recommendations_demo.json
python -m pytest tests/test_policy_recommender.py -v
```

For a real catalogue, provide verified current premium, sum-insured, benefits, waiting-period, exclusion, co-pay, sub-limit and cashless-network facts. The clause parser is a rule-based baseline and must not be treated as authoritative policy/legal interpretation. See [IRDAI health-insurance guidance](https://irdai.gov.in/health-dept).

## 9. Run the OW-ERF v0.2 temporal relational pilot

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install hdbscan
python owerf_v02_benchmark.py --n-samples 2400 --epochs 2 --batch-size 128 --hidden all --tier 2 --seed 42 --output-dir results/owerf_v02_pilot
python -m pytest tests/test_owerf_v02_benchmark.py -v
```

The runner executes four chronological held-out-mechanism folds (one per mechanism). It writes aggregate metrics to `results/owerf_v02_pilot/owerf_v02_synthetic_benchmark.json` and per-fold reports, predictions, generated graph edges and hidden evaluation truth under mechanism-specific directories. The last table must remain separate from model inputs and is for evaluation only.

This is a parametric pilot with causal relation summaries—not a full PyG HGT/R-GCN graph and not an implementation of every experiment in the manuscript. Synthea ingestion and the verified HBP 2.2 mapping remain additional gates.

## Known limitations

- Uses synthetic data only.
- The v0.2 temporal-relational pilot is not a full arbitrary-node PyG HGT/R-GCN implementation.
- The Isolation Forest comparison is a baseline, not a comprehensive benchmark.
- The LSTM/CNN/Transformer and relation-specific attention prototypes are trained on generated synthetic inputs only.
- Results should not be generalized to real-world claims without appropriate datasets and validation.

## 10. Independently train/test LSTM, CNN and Transformer

This benchmark gives each architecture its own classifier rather than relying only on a fused LSTM/CNN/Transformer result.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/train_evaluate_deep_models.py --n-samples 2400 --epochs 5 --batch-size 128 --seeds 42,43,44 --hidden all --output-dir results/standalone_deep_models
python -m pytest tests/test_standalone_deep_models.py -v
```

The run performs 36 fits (three models × four leave-one-mechanism-out folds × three generator seeds). The training, validation, known-only calibration and test windows are separate; the withheld mechanism is excluded from training, validation and calibration. A checkpoint is selected on validation cross-entropy.

Metrics include multiclass known-class accuracy/macro-F1/balanced accuracy (calculated only on test rows from classes seen during training), novel-mechanism AUROC/Average Precision, FPR at 95% novel recall, conformal novel recall and known false-rejection at alpha 0.05, rare-severe legitimate rejection and Precision@K.

Outputs: `results/standalone_deep_models/standalone_deep_model_metrics.json`, `results/standalone_deep_models/standalone_deep_model_metrics.csv`, and weights for all 36 fits under `results/standalone_deep_models/weights/`. The dedicated GitHub Actions workflow validates the run and commits these artifacts on `main`.

