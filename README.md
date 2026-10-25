# MUJ-DS-2430010379

## Repository owner details

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

This repository supports the MUJ-DS training program and the shared OW-ERF team project. It contains the research prototype, paper source, tests, presentation, results, and supporting material, alongside the required training-program folder structure.

The evaluated OW-ERF prototype currently uses **synthetic data only**. Its results are preliminary and must not be interpreted as validation on real insurance claims.

## Repository structure

- `assignments/` — coursework and assignments
- `notebooks/` — exploratory notebooks
- `code/` — project code copies and code-related materials
- `resources/` — screenshots, result files, and supporting resources
- `presentations/` — project presentation
- `capstone/` — space for capstone-related materials within this same repository

**This is the single team-project repository. A separate capstone repository is not required.**

The research prototype and tests are also present at the repository root. See [INSTALLATION.md](INSTALLATION.md) for setup, experiment execution, and test commands.

## Current implementation status

- **Implemented:** synthetic claim generator, tabular OW-ERF pipeline, leave-one-mechanism-out experiment, and Isolation Forest comparison.
- **Automated CI:** GitHub Actions runs the pytest suite on Python 3.11 and 3.12.
- **Latest recorded local test result:** 7 passed and 1 skipped. The skipped tests require optional PyTorch support; check the CI run and the target submission environment before describing test coverage.
- **Not yet validated:** LSTM, CNN, and Transformer branches are written but have not been trained or evaluated.
- **Research limitation:** all reported experimental results are from synthetic data, not real-world insurance claims. The prototype does not implement the proposed graph encoder.

## Team project and research authorship

This is a **team project**, not a solo project. The repository owner's account or the person who commits code does not, by itself, determine the research paper's authorship.

The paper source is [`paper/main.tex`](paper/main.tex). Before formal submission, its author list must be reconciled with the approved project synopsis/team roster and the actual contributions to the research. Each named author should agree to the final author list, and authorship should not be assigned solely because someone is an instructor, supervisor, or team member. The paper source should only be changed after the project team and relevant faculty confirm the correct list.

