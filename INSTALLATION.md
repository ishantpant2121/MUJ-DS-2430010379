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

## 6. Optional deep-learning code

The LSTM, 1D-CNN, and Transformer branches in `owerf_deep_encoders.py` are **written but not trained or evaluated**. The current synthetic generator does not yet provide the sequence inputs needed for a meaningful experiment.

Only if you are intentionally testing this experimental code, install PyTorch using the installation command recommended by the official PyTorch website for your operating system, then run the optional smoke tests from the repository root with `python -m pytest tests/test_deep.py -v`. Do not report the deep branches as working or evaluated until their tests and training experiments have actually completed.

## Known limitations

- Uses synthetic data only.
- Current evaluated pipeline is tabular; it does not implement the proposed graph encoder.
- The Isolation Forest comparison is a baseline, not a comprehensive benchmark.
- The deep-learning branches are not trained or evaluated.
- Results should not be generalized to real-world claims without appropriate datasets and validation.
