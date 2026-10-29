# OW-ERF / Insurance Intelligence — AI Handover and Roadmap

**Owner:** Ishant Pant, Manipal University Jaipur  
**Repository:** https://github.com/ishantpant2121/MUJ-DS-2430010379  
**Updated:** 2026-10-29  
**Purpose:** Transfer project context, current implementation, known evidence, release status and next steps to another AI or developer.

## 1. Executive status

The project is a research prototype named **OW-ERF (Open-World Explained Rarity Filtering)** for health-insurance claim intelligence. It separates known-class classification from novel-mechanism detection and uses clinical/structural context to decide which unusual cases merit human review.

**Do not describe the project as fully research-complete or production-ready.** The core prototype, controlled standalone neural-network comparison, research paper, presentation decks, a custom temporal heterogeneous graph encoder, data-ingestion tooling and a submission checklist exist. The main controlled neural experiment is complete and synthetic-only. The temporal graph encoder has been merged to `main`, and CI smoke/unit tests were reported green in the prior work session; however, the complete 4-held-out-mechanism × 3-seed graph benchmark has not been reviewed as a completed empirical result. Real labelled claim-level validation, final external-source verification, package-rate curation, and submission administration remain outstanding.

## 2. Repository and important artifacts

Repository: https://github.com/ishantpant2121/MUJ-DS-2430010379

Key code:
- `owerf_v02_benchmark.py`: OW-ERF v0.2 synthetic benchmark, novelty scoring and explained-rarity gate.
- `owerf_temporal_heterograph.py`: custom typed-node temporal graph encoder.
- `scripts/train_evaluate_deep_models.py`: standalone LSTM, 1D-CNN and Transformer experiment.
- `scripts/summarize_deep_comparison.py`: controlled deep-model comparison report generation.
- `scripts/train_temporal_heterograph.py`: graph model train/evaluation pipeline.
- `scripts/prepare_synthea_graph.py`: Synthea CSV-to-typed-graph adapter with source-file hashes and a manifest. It does not train a fraud classifier and explicitly reports that fraud labels are absent.
- `scripts/validate_hbp_catalog.py`: structural/provenance checks for a manually reviewed HBP package catalogue. It cannot independently prove that a transcribed rate matches the official PDF.
- `data/hbp_package_catalog_template.csv`: empty, provenance-aware HBP catalogue template; no official rates are fabricated.
- `tests/test_temporal_heterograph.py`: graph schema, gradients, strict-past policy and future-feature invariance tests.
- `tests/test_data_ingestion.py`: Synthea adapter and HBP provenance-validator tests.
- `.github/workflows/temporal-heterograph.yml`: CI smoke run and manually selectable full graph benchmark.
- `docs/TEMPORAL_HETEROGRAPH_MODEL.md`: graph design, temporal policy, run commands and caveats.
- `docs/FINAL_SUBMISSION_CHECKLIST.md`: final submission checklist.
- `docs/OW_ERF_V02_IMPLEMENTATION_MAP.md`: scope and implementation map.
- `paper/main.tex`: LaTeX paper source.
- `paper/OWERF_Research_Paper.pdf`: generated paper PDF, if the latest PDF build has completed successfully.

Previously created editable decks in the earlier assistant session:
- `OW_ERF_Combined_Presentation_Final_60ep.pptx`
- `Insurance_Intelligence_PRD_Final_60ep.pptx`

These were generated and visually checked as local session downloads, but were **not confirmed committed to the GitHub repository**. Re-create or retrieve the files from the original conversation if they are needed; do not assume they exist in the repo.

## 3. What has been implemented

### OW-ERF baseline
- Known-class classification separated from open-world novelty scoring.
- Structural anomaly and clinical expectation/inconsistency components.
- Explained-rarity escalation gate intended to avoid escalating every rare claim.
- Synthetic evaluation with held-out mechanisms.
- Tests for the conjunctive gate and past-only relation-count policy were added in the earlier work.

The gate is a human-review triage heuristic, not proof of fraud and not an automated denial rule.

### Controlled deep-model comparison
- LSTM, continuous-input 1D-CNN and Transformer compared under a common standardized relational-context tensor and padding mask.
- Train-only scaling and explicit padding behavior.
- Native-input results retained separately to avoid claiming the original input settings were controlled.
- 60 epochs maximum, three generator seeds (42, 43, 44), four held-out mechanisms (upcoding, unbundling, phantom billing, collusion), and 12 held-out-mechanism/seed runs per architecture.
- The controlled workflow and the Python 3.11/3.12 test matrix passed in the earlier session.

### Shared-context 60-epoch results (synthetic data only)
Mean ± SD over 12 runs per architecture:
- LSTM: known accuracy 0.979 ± 0.010; macro-F1 0.908 ± 0.054; novelty AUROC 0.721 ± 0.210; AP 0.169 ± 0.105; novel recall at nominal alpha 0.05 = 0.250 ± 0.166; known false-rejection 0.048 ± 0.019; rare-severe benign false-rejection 0.033 ± 0.037.
- 1D-CNN: accuracy 0.973 ± 0.016; macro-F1 0.865 ± 0.091; AUROC 0.759 ± 0.171; AP 0.188 ± 0.141; novel recall 0.354 ± 0.325; known false-rejection 0.054 ± 0.015; rare-severe benign false-rejection 0.035 ± 0.041.
- Transformer: accuracy 0.983 ± 0.009; macro-F1 0.926 ± 0.042; AUROC 0.816 ± 0.090; AP 0.174 ± 0.103; novel recall 0.349 ± 0.228; known false-rejection 0.061 ± 0.021; rare-severe benign false-rejection 0.040 ± 0.040.

Interpretation: the Transformer had the strongest mean known-class and novelty-AUROC scores in this particular synthetic experiment, but novelty results vary substantially. Do not claim universal superiority or real-world effectiveness. In native-input mode, LSTM/CNN were materially affected by input representation; Transformer already used the shared context.

Existing result files:
- `results/controlled_deep_model_comparison/controlled_comparison_summary_60ep.json`
- `results/controlled_deep_model_comparison/native_60ep/standalone_deep_model_metrics.json`
- `results/controlled_deep_model_comparison/shared_context_60ep/standalone_deep_model_metrics.json`
- Earlier 25-epoch pilot remains separately preserved at `results/controlled_deep_model_comparison/controlled_comparison_summary.json`.

### Temporal heterogeneous graph model
- Seven explicit node types: Claim, Beneficiary, Hospital, Doctor, Admission, Package, Diagnosis.
- Six typed relation channels, relation-specific transformations, recency/time features, historical entity messages, node-type embeddings, relation attention, known-class output and expectation head.
- Conservative temporal snapshots: for query block [start, end), only events with index < start may contribute to historical entity messages; claims in the current query block do not message into entities used by other claims in that block.
- Chronological split: train 55%, model selection 10%, calibration 15%, deployment evaluation 20%; train-only feature scaling and known-only calibration.
- It is a **custom dependency-light PyTorch R-GCN/HGT-style implementation**, not canonical PyTorch Geometric `HGTConv` or `RGCNConv`.
- Synthetic Admission is one node per claim; Diagnosis is a severity-bin proxy. Replace these with defensible entities only when source data support them.
- Smoke tests verify execution and core invariants; smoke success is not a substitute for the full benchmark.

## 4. Paper and PDF state

The paper reports the completed 60-epoch standalone deep-model experiment and clearly states that graph-model numerical results are not yet included because the full graph benchmark has not been reviewed. It describes the custom graph architecture and its synthetic proxy nodes. Do not add graph-model performance claims until the full benchmark artifacts have been generated and audited.

A LaTeX escaping issue for the filename `owerf_temporal_heterograph.py` was identified and corrected on `main` in the prior session. **Verify the latest paper build after the latest main commit** before treating the PDF as final:
- Actions: https://github.com/ishantpant2121/MUJ-DS-2430010379/actions
- PDF path: https://github.com/ishantpant2121/MUJ-DS-2430010379/blob/main/paper/OWERF_Research_Paper.pdf

Do not claim a build passed unless the latest workflow run for the current main commit has a successful conclusion.

## 5. Roadmap — do these in order

### P0 — Validate current main and finish the graph benchmark
1. Inspect latest commit and all GitHub Actions runs. Confirm graph unit tests, data-ingestion tests, existing Python 3.11/3.12 tests and paper PDF build are green on the latest `main`.
2. Run the workflow `Temporal heterogeneous graph benchmark` manually with `full_benchmark=true`, or run locally:
   ```bash
   python -m pip install -r requirements.txt
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements-deep.txt
   python -m pytest -q tests/test_temporal_heterograph.py tests/test_data_ingestion.py
   python scripts/train_temporal_heterograph.py --n-samples 2400 --epochs 8 --batch-size 64 --patience 3 --seeds 42,43,44 --hidden all --output-dir results/temporal_heterograph_full
   ```
3. Confirm four held-out mechanism folds × three seeds actually ran, and verify per-fold reports, predictions, checkpoints, aggregate JSON/CSV and seed/fold metadata.
4. Inspect for bugs, leakage, failed folds, empty/undefined metrics, class-count errors, and calibration issues. Record mean ± SD and mechanism-specific results.
5. Compare against the existing tabular OW-ERF baseline, relation-summary baseline, and standalone deep models using identical splits and comparable inputs where feasible.
6. Update the paper only with verified results. Regenerate the PDF and inspect it visually.

### P1 — Ablations and robustness
1. Ablate temporal recency weighting, relation-specific transforms, relation attention, expectation head, structural novelty score, and explained-rarity gate.
2. Evaluate cold-start and rare-entity behavior, thresholds/calibration, false rejection among rare/severe benign claims, and escalation workload.
3. Increase seeds and report per-mechanism distributions, confidence intervals where appropriate, and denominators.
4. Audit shortcuts: IDs, timestamps, generator artifacts, entity degree, proxy nodes, and hidden-mechanism information must not leak into features.
5. Evaluate provider/entity-group error patterns and explanation stability. Treat any subgroup analysis as research and use lawful, justified group definitions.

### P2 — External data pipeline (not fraud validation by itself)
1. Pin a Synthea release/commit and exporter settings.
2. Run `scripts/prepare_synthea_graph.py --input-dir <synthea-csv-dir> --output-dir data/prepared_synthea_graph`.
3. Inspect `manifest.json`, input hashes, node/edge counts, timestamps, missing identifiers, duplicate codes, and temporal consistency.
4. Synthea is synthetic clinical data and does not provide ground-truth insurance fraud mechanisms. Use it for schema/ingestion/clinical-trajectory checks only unless a separately justified label design exists.
5. Manually curate a small HBP 2.2 package-rate subset only if required. Record exact package code/name, tier, INR rate, source document, source page, reviewer and verification date. Run `scripts/validate_hbp_catalog.py`; this checks structure/provenance fields, not source truth. A human must compare every row to the official source.
6. Never invent official HBP rates, claim that Synthea is fraud-labelled, or present synthetic data as real claims.

### P3 — Proper labelled evaluation
1. Seek appropriately governed, anonymized, claim-level data with defensible labels and permissions.
2. Design temporal and provider-held-out evaluation; define label quality, fraud prevalence, and novel mechanism splits before training.
3. Evaluate at realistic prevalence and report precision-recall, recall at fixed false-positive/false-rejection rates, calibration, investigation volume, and cost-weighted utility.
4. Keep human review, appeal and audit logs. Do not automate claim denial from this prototype.

### P4 — Final submission/admin
1. Verify every citation against the original source; validate bibliographic metadata and links.
2. Confirm course rubric, exact submission requirements, AI-assistance disclosure, and whether editable PPTX decks are required.
3. If decks are required in GitHub, upload the final editable `.pptx` files under `presentations/`; earlier copies were session downloads and were not confirmed committed.
4. Confirm whether the instructor's collaborator invitation is pending and accept it if required.
5. Sync external Overleaf only if still in use.
6. Export final paper/decks from the same revision, visually inspect every page/slide, and confirm all links/downloads.
7. Keep `docs/FINAL_SUBMISSION_CHECKLIST.md` current.

## 6. Important GitHub links

- Repository: https://github.com/ishantpant2121/MUJ-DS-2430010379
- Actions: https://github.com/ishantpant2121/MUJ-DS-2430010379/actions
- Paper source: https://github.com/ishantpant2121/MUJ-DS-2430010379/blob/main/paper/main.tex
- Paper PDF: https://github.com/ishantpant2121/MUJ-DS-2430010379/blob/main/paper/OWERF_Research_Paper.pdf
- Graph model: https://github.com/ishantpant2121/MUJ-DS-2430010379/blob/main/owerf_temporal_heterograph.py
- Graph training script: https://github.com/ishantpant2121/MUJ-DS-2430010379/blob/main/scripts/train_temporal_heterograph.py
- Graph design notes: https://github.com/ishantpant2121/MUJ-DS-2430010379/blob/main/docs/TEMPORAL_HETEROGRAPH_MODEL.md
- Submission checklist: https://github.com/ishantpant2121/MUJ-DS-2430010379/blob/main/docs/FINAL_SUBMISSION_CHECKLIST.md

## 7. Instructions to the next AI

Start by reading this handover, the README, `docs/FINAL_SUBMISSION_CHECKLIST.md`, `docs/TEMPORAL_HETEROGRAPH_MODEL.md`, `docs/OW_ERF_V02_IMPLEMENTATION_MAP.md`, `paper/main.tex`, the latest Actions logs, and the actual training scripts. Verify current repository state rather than relying on this note blindly. Do not rewrite results from memory; use committed JSON/CSV artifacts. Do not silently change the paper's claims, fabricate data, invent HBP rates, or claim real-world validation. Make focused commits, run relevant tests, and report exact commit SHAs and workflow URLs.
