# Insurance Intelligence & OW-ERF

**An explainable health-insurance policy recommendation prototype and an open-world fraud-triage research framework.**

[![Python tests](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/workflows/tests.yml/badge.svg)](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/workflows/tests.yml)
[![Temporal graph benchmark](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/workflows/temporal-heterograph.yml/badge.svg)](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/workflows/temporal-heterograph.yml)
[![Paper build](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/workflows/build-paper.yml/badge.svg)](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/workflows/build-paper.yml)

**Author:** Ishant Pant · Data Science · Manipal University Jaipur  
**Research supervisor:** Dr. Sandeep Gupta

---

## Overview

This repository contains two related but distinct components:

1. **Insurance policy recommendation** — a transparent catalogue-ranking prototype that explains policy/profile compatibility and surfaces missing benefits and cautions.
2. **OW-ERF (Open-World Explained-Rarity Filtering)** — a research pipeline for flagging unfamiliar claims for human audit when rarity is not adequately explained by clinical or relational context.

The repository includes tabular, deep-learning, temporal relation-summary and custom typed-node temporal-graph experiments, a separate public **provider-level** benchmark, an interactive Streamlit application, tests, experiment reports and a LaTeX research manuscript.

> **Research status:** claim-mechanism experiments use a parametric synthetic generator. They do not establish real-world claim-level fraud performance. The public dataset experiment predicts a provider-level `PotentialFraud` label and is not claim-level mechanism ground truth. Outputs are for research and human audit triage, not automated adverse decisions.

## Start here

| Resource | Description |
|---|---|
| [Research paper (PDF)](paper/OWERF_Research_Paper.pdf) | Compiled manuscript currently tracked in the repository |
| [LaTeX source](paper/main.tex) · [Bibliography](paper/references.bib) | Source files for Overleaf or local compilation |
| [Installation and run guide](INSTALLATION.md) | Environment setup and experiment commands |
| [Project handover](docs/PROJECT_HANDOVER_FOR_NEXT_AI.md) | Architecture, completed work and roadmap |
| [Final submission checklist](docs/FINAL_SUBMISSION_CHECKLIST.md) | Research and administrative items still to verify |
| [Temporal graph result analysis](docs/TEMPORAL_GRAPH_BENCHMARK_RESULTS.md) | Matched graph/no-graph results and interpretation |
| [OW-ERF implementation map](docs/OW_ERF_V02_IMPLEMENTATION_MAP.md) | Method stages mapped to actual code |
| [Deep-model guide](docs/DEEP_LEARNING_MODELS.md) | Architectures, metrics and evaluation protocol |
| [Dataset feasibility review](DATASET_REVIEW.md) | Dataset roles and label limitations |
| [Product requirements](docs/INSURANCE_INTELLIGENCE_PRD.md) | App goals, requirements and guardrails |

## Architecture at a glance

- **Application:** `app.py` (Streamlit)
- **Policy ranking:** `policy_recommender.py`
- **Tabular OW-ERF baseline:** `owerf_core.py`
- **Temporal relation-summary pilot:** `owerf_v02_benchmark.py`
- **Deep encoders and experiment:** `owerf_deep_encoders.py`, `owerf_deep_experiment.py`
- **Standalone LSTM / 1D-CNN / Transformer evaluation:** `scripts/train_evaluate_deep_models.py`
- **Custom typed-node temporal graph:** `owerf_temporal_heterograph.py`, trained by `scripts/train_temporal_heterograph.py`
- **Synthea CSV graph adapter:** `scripts/prepare_synthea_graph.py`
- **HBP provenance validator:** `scripts/validate_hbp_catalog.py`
- **Provider-level public benchmark:** `scripts/train_public_provider_dataset.py`
- **Automated tests:** `tests/`
- **Reproducible results:** `results/`

The temporal graph model is a custom PyTorch R-GCN/HGT-style message-passing implementation, **not** canonical PyTorch Geometric `HGTConv`/`RGCNConv`. Its current Admission and Diagnosis nodes are synthetic proxies. See the model documentation for exact assumptions.

## Quick start

Python 3.10+ is recommended. From the repository root:

```bash
git clone https://github.com/ishantpant2121/MUJ-DS-2430010379.git
cd MUJ-DS-2430010379

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-app.txt
streamlit run app.py
```

For PyTorch-based experiments, install CPU PyTorch and the deep-model requirements as needed:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-deep.txt
python -m pytest -v
```

For detailed platform-specific setup and individual commands, use [INSTALLATION.md](INSTALLATION.md).

### Run the policy recommendation demo

```bash
python policy_recommender.py \
  --profile data/demo_customer_profile.json \
  --catalog data/demo_policy_catalog.json \
  --top-k 3 \
  --output results/policy_recommendations_demo.json

python -m pytest tests/test_policy_recommender.py -v
```

The bundled catalogue and customer profile are fictional fixtures, not real insurance products or purchasing advice.

### Run the temporal graph experiment

```bash
python scripts/train_temporal_heterograph.py \
  --n-samples 2400 --epochs 8 --batch-size 64 --patience 3 \
  --seeds 42,43,44 --hidden all \
  --output-dir results/temporal_heterograph_full
```

For the matched relation-free control, add `--ablation no_graph` and use a separate output directory. The GitHub Actions workflow also provides a smoke test and a manually triggered full synthetic benchmark.

## Key research result: temporal graph vs. no-graph control

The completed matched experiment uses four held-out mechanisms (upcoding, unbundling, phantom billing and collusion), three seeds (42, 43, 44), 2,400 synthetic claims per run and an eight-epoch cap: **12 runs per arm**.

| Metric (mean ± SD across 12 runs) | Temporal graph | No-graph control |
|---|---:|---:|
| Known-class accuracy | 0.9566 ± 0.0111 | 0.9590 ± 0.0114 |
| Known macro-F1 | 0.6516 ± 0.0946 | 0.6655 ± 0.0998 |
| Novel-mechanism AUROC | 0.6965 ± 0.1662 | 0.7062 ± 0.1895 |
| Novel average precision | 0.0899 ± 0.0687 | 0.0980 ± 0.0741 |
| Novel recall at nominal α = 0.05 | 0.1077 ± 0.1263 | 0.1272 ± 0.1273 |

**Interpretation:** the current synthetic experiment does not demonstrate a benefit from graph message passing; the no-graph control is slightly better on the main novelty metrics. Novel recall is low and variable, and collusion detection is near/below chance in the graph arm. This negative result is documented rather than hidden. Full details: [benchmark report](docs/TEMPORAL_GRAPH_BENCHMARK_RESULTS.md) and [machine-readable summary](results/temporal_heterograph_comparison_summary.json).

A separate controlled 60-epoch LSTM/CNN/Transformer experiment gives all three models the same standardized relational-context input. Its complete table and caveats are in [Deep-model guide](docs/DEEP_LEARNING_MODELS.md) and [60-epoch summary](results/controlled_deep_model_comparison/controlled_comparison_summary_60ep.json).

## Data and evaluation boundaries

- **Synthetic claim data:** used for controlled held-out-mechanism experiments. These scores are exploratory and do not generalize automatically to real claims.
- **Public provider dataset:** its `PotentialFraud` target is provider-level. It is a separate supervised benchmark, not ground-truth labels for individual claim mechanisms. See [dataset review](DATASET_REVIEW.md) and [provider report](results/public_provider_fraud_report.json).
- **Synthea:** the adapter creates typed nodes, timestamped edges and a source hash manifest. Synthea is synthetic clinical data and does not provide fraud-mechanism labels by default.
- **HBP package rates:** the repository contains a provenance template and validator, not an automatically verified official rate catalogue. Human-check every transcribed rate against the relevant official source page.
- **Policy recommendation:** the demo uses fictional policies and a heuristic clause parser. It is not legal interpretation or a substitute for official policy wording.

## Reproducibility and CI

GitHub Actions workflows cover standard Python tests, the paper PDF build, deep-model experiments, the provider-level benchmark and temporal graph runs. Check the workflow for the commit and artifacts you intend to cite; do not treat a successful smoke test as a completed full experiment.

The tracked result files under `results/` are the compact, reviewable summaries. Detailed fold predictions and model checkpoints may be attached to workflow artifacts; download and archive any artifact required for long-term reproducibility.

## Overleaf

The manuscript source is split into two files:

- [`paper/main.tex`](paper/main.tex) — manuscript
- [`paper/references.bib`](paper/references.bib) — BibTeX bibliography

To use Overleaf, create a blank project, upload both files, set `main.tex` as the main document and compile with pdfLaTeX/BibTeX (or use Overleaf's automatic bibliography handling). The current manuscript can be viewed in GitHub; when you provide your revised paper, replace the manuscript source and rebuild the PDF only after checking that the updated paper and bibliography compile together.

## Repository conventions

- Keep source code, tests, documentation and compact canonical results in version control.
- Do not commit downloaded raw external datasets, local virtual environments, caches, or transient training outputs.
- Preserve older benchmark reports only when they represent a distinct protocol (for example, the 25-epoch pilot versus the 60-epoch controlled run).
- Treat workflow artifacts as temporary unless explicitly archived.
- Do not delete checkpoints/results until confirming they are not the only saved artifacts for a distinct experiment.

## License and responsible use

No open-source license is declared unless a license file is added. Before redistributing or deploying, review the upstream dataset licenses and add an appropriate project license. This is an academic research prototype; any operational use would require approved real-world data, rigorous external validation, privacy/security review and human oversight.
