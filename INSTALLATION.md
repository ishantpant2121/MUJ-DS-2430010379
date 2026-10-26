# OW-ERF Installation and Run Guide

This guide runs the current **tabular synthetic-data prototype**. It does not install or train the experimental deep-learning branches.

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

The last recorded test run reported **7 passed, 1 skipped**. The skip is associated with the optional PyTorch tests; this status should be rechecked in the environment used for submission.

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

## Known limitations

- Uses synthetic data only.
- Current evaluated pipeline is tabular; it does not implement the proposed graph encoder.
- The Isolation Forest comparison is a baseline, not a comprehensive benchmark.
- The deep-learning branches are not trained or evaluated.
- Results should not be generalized to real-world claims without appropriate datasets and validation.
