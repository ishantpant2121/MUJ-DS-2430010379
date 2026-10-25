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

This repository contains the **individual OW-ERF research project led and maintained by Ishant Pant**. There are no other student team members. The manuscript lists Dr. Sandeep Gupta as research supervisor and second author.

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

## Current implementation status

- **Implemented and evaluated:** synthetic claim generator, tabular OW-ERF pipeline, leave-one-mechanism-out experiment, and Isolation Forest comparison.
- **Automated CI:** GitHub Actions runs the pytest suite on Python 3.11 and 3.12.
- **Deep-learning branches:** LSTM, 1D-CNN, and Transformer code is present but has not been trained or evaluated.
- **Research limitations:** all reported experimental results use synthetic data; the evaluated prototype is tabular and does not implement the proposed graph encoder. Results should not be generalized to real-world claims without appropriate data and validation.

## Research authorship

- **Ishant Pant** — lead / first author.
- **Dr. Sandeep Gupta** — research supervisor and second author.

Bibliography entries retain the names of researchers whose publications are cited; those citations credit external research and do not identify additional project team members.
