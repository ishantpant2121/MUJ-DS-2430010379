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

This is my personal repository for the MUJ-DS training program. It contains the required folder structure and project-related materials. The OW-ERF research prototype currently uses **synthetic data**; its results should not be interpreted as validation on real insurance claims.

## Repository structure

- `assignments/` — coursework and assignments
- `notebooks/` — exploratory notebooks
- `code/` — project code copies and code-related materials
- `resources/` — screenshots, result files, and supporting resources
- `presentations/` — project presentation
- `capstone/` — capstone-related materials

The research prototype and its tests are also present at the repository root. Until the capstone repository is created and verified, this repository should not be assumed to be a separate capstone repository.

## Run the current tabular prototype

See [INSTALLATION.md](INSTALLATION.md) for environment setup, experiment execution, and tests.

## Current implementation status

- **Implemented:** synthetic claim generator, tabular OW-ERF pipeline, leave-one-mechanism-out experiment, and Isolation Forest comparison.
- **Tests:** the latest recorded run reported 7 passed and 1 skipped. The skipped tests require PyTorch.
- **Not yet validated:** LSTM, CNN, and Transformer branches are written but have not been trained or evaluated.
- **Research limitation:** all reported experimental results are from synthetic data, not real-world insurance claims.

## Capstone repository

**Status: separate capstone repository not yet verified.** The previous URL pointed to `ow-erf-fraud-detection`, but a separate repository has not been confirmed through the connected GitHub account. Replace this section with the verified capstone URL after the repository is created.

## Individual contributions

This is currently documented as a solo project. Contributions include problem formulation, synthetic data generation, model implementation, testing, documentation, paper preparation, and presentation preparation. AI assistance was used and is disclosed in the research materials.
