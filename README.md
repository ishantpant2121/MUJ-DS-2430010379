# MUJ-DS-2430010379

## Student details

| Field | Details |
|---|---|
| **Name** | Ishant Pant |
| **Registration Number** | 2430010379 |
| **Branch** | Data Science |
| **Batch** | D |
| **Project Title** | Open-World Explained-Rarity Filtering for Health-Insurance Claim Fraud (OW-ERF) |
| **GitHub Username** | [ishantpant2121](https://github.com/ishantpant2121) |
| **Training Program** | MUJ-DS Training Program, Batch D |

## About this repository

This repository contains the **individual OW-ERF research project led and maintained by Ishant Pant**. There are no other student team members. The manuscript lists Dr. Sandeep Gupta as research supervisor.

The evaluated OW-ERF prototype currently uses **synthetic data only**. Its preliminary results must not be interpreted as validation on real insurance claims.

## Repository structure

- `.github/workflows/` — automated Python tests and research-paper PDF build
- `paper/` — LaTeX manuscript, bibliography, and compiled research-paper PDF
- `presentations/` — project presentation
- `resources/` — screenshots of experiment/test runs
- `results/` — canonical recorded experiment results
- `tests/` — tests for the tabular prototype and optional deep-learning code
- `owerf_core.py` — evaluated tabular prototype and experiment entry point
- `owerf_deep_encoders.py` — experimental LSTM, 1D-CNN, and Transformer branches
- `requirements.txt` — dependencies for the evaluated tabular prototype
- `requirements-deep.txt` — optional dependencies for experimental deep-learning branches
- `INSTALLATION.md` — setup, run, and test instructions

Empty placeholder directories and duplicate copies of source/results files have been removed. The project does not require a separate capstone repository.

## Research paper

[**View or download the compiled research paper PDF**](paper/OWERF_Research_Paper.pdf)

The PDF is built from `paper/main.tex` and `paper/references.bib`. GitHub Actions rebuilds the PDF when the manuscript, bibliography, or build workflow changes.

## Dataset feasibility

[Public dataset feasibility review](DATASET_REVIEW.md) documents potential CMS Medicare and OIG exclusion-list sources and why these provider-level data do not directly validate claim-level discovery of the four synthetic fraud mechanisms. No external dataset experiment has been run yet.

## Run the deep prototype

Install the base dependencies and CPU PyTorch, then run a synthetic hidden-mechanism experiment:

```bash
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python owerf_deep_experiment.py --n-samples 4000 --epochs 12 --hidden collusion --seed 42
```

The command trains on three known synthetic fraud mechanisms, calibrates on a separate known-class split, tests on the held-out mechanism, applies the explained-rarity gate, compares novelty ranking with Isolation Forest, and writes `results/deep_prototype_results.json`. Use `--hidden upcoding`, `--hidden unbundling`, or `--hidden phantom_billing` to test another held-out mechanism. This is a proof-of-pipeline on generated data, not external validation or a complete temporal graph model.

## Current implementation status

- **Implemented and evaluated:** synthetic claim generator, tabular OW-ERF pipeline, leave-one-mechanism-out experiment, and Isolation Forest comparison.
- **Automated CI:** GitHub Actions runs the pytest suite on Python 3.11 and 3.12.
- **Deep prototype:** A runnable synthetic-data training/evaluation pipeline trains the LSTM, 1D-CNN, and Transformer branches, calibrates novelty, applies explained-rarity filtering, and compares with Isolation Forest. It is not real-claims validation.
- **Research limitations:** all reported experimental results use synthetic data; the evaluated prototype is tabular and does not implement the proposed graph encoder. Results should not be generalized to real-world claims without appropriate data and validation.

## Research authorship

- **Ishant Pant** — lead / first author.
- **Dr. Sandeep Gupta** — research supervisor.

Bibliography entries retain the names of researchers whose publications are cited; those citations credit external research and do not identify additional project team members.
