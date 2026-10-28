# Project Handover for the Next AI — Insurance Intelligence / OW-ERF

**Repository:** https://github.com/ishantpant2121/MUJ-DS-2430010379  
**Branch:** main  
**Handover date:** 28 October 2026  
**Scope:** Individual B.Tech Data Science project combining customer policy recommendation with research on open-world health-insurance claim fraud triage.

## Executive status

The code, graph experiment, matched ablation, documentation and manuscript updates have been committed to GitHub main. The standard Python test matrix has passed on Python 3.11 and 3.12 on the latest verified revisions. The temporal graph smoke and full graph/no-graph workflows passed; the full run artifacts are retained in GitHub Actions. The paper PDF was successfully rebuilt after the paper text was updated; a later bibliography metadata update has started another paper build, so verify its latest result before claiming the final PDF is current.

This is **not** a fully validated, real-world fraud-detection system. All claim-level fraud-mechanism results are parametric synthetic experiments. The most recent matched ablation did not show a benefit from graph message passing: the no-graph control was slightly better on mean novelty AUROC/AP/recall. The project should not claim real-world claim-level performance or that graph context improved detection.

## What is in main

### Application and earlier research
- `app.py`: Streamlit application prototype.
- `policy_recommender.py`: customer policy compatibility/ranking prototype.
- `owerf_core.py`: tabular OW-ERF prototype.
- `owerf_v02_benchmark.py`: temporal relation-summary pilot, including past-only entity summaries, novelty score fusion, conformal rejection, explained-rarity gate, clustering/audit summaries.
- `owerf_deep_encoders.py` and `owerf_deep_experiment.py`: earlier fused deep-model prototype.
- `scripts/train_evaluate_deep_models.py`: standalone LSTM/CNN/Transformer synthetic benchmark.
- `results/controlled_deep_model_comparison/`: controlled 60-epoch shared-input comparison and model checkpoints/results.

### Custom temporal graph baseline
- `owerf_temporal_heterograph.py`: custom core-PyTorch temporal typed-node model. It has eight node types: Claim, Beneficiary, Hospital, Doctor, Provider, Admission, Package and Diagnosis. It uses eight relation channels: provider, hospital, doctor, beneficiary, admission, package, diagnosis and referral target. Provider and referral-target channels share the Provider node namespace but use separate relation-specific projections.
- `scripts/train_temporal_heterograph.py`: chronological train/selection/calibration/deployment training and evaluation pipeline.
- `tests/test_temporal_heterograph.py`: schema, gradient, temporal guard and future-feature invariance tests.
- `.github/workflows/temporal-heterograph.yml`: CPU smoke benchmark and manually selectable full synthetic benchmark.
- The encoder is **custom R-GCN/HGT-style PyTorch message passing, not canonical PyG HGTConv/RGCNConv**. Admission is one synthetic node per claim; Diagnosis is a severity-bin proxy. It is a research baseline, not a validated deployment model.

### Data preparation / provenance
- `scripts/prepare_synthea_graph.py`: Synthea CSV to typed nodes/timestamped edges with input SHA-256 manifest. Synthea does not provide claim-level fraud labels; this adapter does not produce fraud model scores.
- `data/hbp_package_catalog_template.csv` and `scripts/validate_hbp_catalog.py`: source-provenance template/validator for manually transcribed HBP package rates. No official rates are invented or prefilled.
- `tests/test_data_ingestion.py`: adapter/provenance tests.

### Research documentation and deliverables
- `paper/main.tex`, `paper/references.bib`, `paper/OWERF_Research_Paper.pdf`.
- `docs/TEMPORAL_HETEROGRAPH_MODEL.md`: model/protocol documentation.
- `docs/TEMPORAL_GRAPH_BENCHMARK_RESULTS.md`: current paired results and limitations.
- `docs/FINAL_SUBMISSION_CHECKLIST.md`: final release checklist.
- `docs/OW_ERF_V02_IMPLEMENTATION_MAP.md`: design-to-code implementation map.
- `docs/DEEP_LEARNING_MODELS.md`, `docs/INSURANCE_INTELLIGENCE_PRD.md`, `docs/OW_ERF_V02_DATA_CARD.md`.
- The editable slide decks were prepared as session downloads but are **not committed as PPTX binaries** in GitHub. Upload them under `presentations/` only if the course requires that.

## Latest full graph vs matched no-graph synthetic experiment

Successful workflow with full graph and matched control artifacts:
https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40680366943

Protocol: four held-out mechanisms (upcoding, unbundling, phantom billing, collusion) × three generator seeds (42, 43, 44) = 12 runs per arm; 2,400 generated claims per run; CPU; eight-epoch cap.

| Metric (mean ± SD over 12 runs) | Full graph | No-graph control |
|---|---:|---:|
| Known-class accuracy | 0.9566 ± 0.0111 | 0.9590 ± 0.0114 |
| Known macro-F1 | 0.6516 ± 0.0946 | 0.6655 ± 0.0998 |
| Novel AUROC | 0.6965 ± 0.1662 | 0.7062 ± 0.1895 |
| Novel average precision | 0.0899 ± 0.0687 | 0.0980 ± 0.0741 |
| Novel recall at alpha = 0.05 | 0.1077 ± 0.1263 | 0.1272 ± 0.1273 |
| Known false-rejection rate | 0.0391 ± 0.0132 | 0.0380 ± 0.0118 |
| Rare-severe benign false-rejection | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Gated novel recall | 0.1077 ± 0.1263 | 0.1212 ± 0.1245 |
| Escalation rate | 0.0401 ± 0.0131 | 0.0396 ± 0.0115 |

Interpretation:
- The no-graph control is slightly ahead on the mean classification and novelty metrics. The current simulator does not demonstrate benefit from the graph branch.
- Full-graph AUROC/recall vary by mechanism: upcoding AUROC 0.7510 ± 0.0164 / recall 0.1505 ± 0.0655; unbundling 0.6876 ± 0.0221 / recall 0; phantom billing 0.8920 ± 0.0192 / recall 0.2545 ± 0.1376; collusion 0.4553 ± 0.0404 / recall 0.0256 ± 0.0444.
- Collusion AUROC is below chance in the full-graph arm. Novel recall is low and variable.
- The explained-rarity gate does not improve full-graph novel recall in this study.
- These are synthetic results and not real insurance claim performance or a clinical safety guarantee.

The aggregate graph results are tracked in:
- `results/temporal_heterograph_full/temporal_heterograph_metrics_summary.json`
- `results/temporal_heterograph_full/temporal_heterograph_metrics_summary.csv`
- `results/temporal_heterograph_full/temporal_heterograph_by_mechanism.csv`
- `results/temporal_heterograph_no_graph/`
- `results/temporal_heterograph_comparison_summary.csv`
- `results/temporal_heterograph_comparison_summary.json`

Full fold reports, test predictions and model checkpoints are in the Actions artifacts linked above. Artifacts are temporary; download/archive them for long-term reproducibility.

## Reproducible commands

From repository root:

```bash
python -m pip install -r requirements.txt -r requirements-deep.txt
python -m pytest -v
python -m pytest -q tests/test_temporal_heterograph.py tests/test_data_ingestion.py
python scripts/train_temporal_heterograph.py --n-samples 2400 --epochs 8 --batch-size 64 --patience 3 --seeds 42,43,44 --hidden all --output-dir results/temporal_heterograph_full
python scripts/train_temporal_heterograph.py --n-samples 2400 --epochs 8 --batch-size 64 --patience 3 --seeds 42,43,44 --hidden all --ablation no_graph --output-dir results/temporal_heterograph_no_graph
```

The complete model workflow and artifacts are preferable to re-running on constrained hardware. The full 12-run arms are not a substitute for the broader pre-registered baseline/ablation protocol.

## GitHub status and cleanup

- PR #32 added the custom typed-node temporal graph baseline and was merged.
- PR #36 added the matched relation-free control and was merged.
- Superseded, stale setup PR #18 was closed, not merged.
- The current main branch has no open PRs.
- A repository audit found no zero-byte tracked files, `.gitkeep`/ `.keep` placeholders, or tracked empty directories. Git does not track empty directories.
- Current Python test workflow on main passed on Python 3.11 and Python 3.12 before this handover document; check the latest Actions status again after the handover commit.
- The paper PDF build passed after the paper-source update. The bibliography metadata was subsequently improved, which triggers another PDF build. Verify the newest build before treating the currently committed PDF as the final matching export.

## Next roadmap — do this in order

1. **Verify current CI/PDF after the latest commit.** Fix any failures first. Confirm latest main SHA and ensure the PDF was built from current `paper/main.tex` + `paper/references.bib`.
2. **Citation/source audit.** BibTeX syntax and key coverage are structurally checked; still verify each substantive manuscript claim against the original source and inspect DOI/URL resolution. Do not mark this complete until a source-by-source audit exists.
3. **Fix the negative graph result scientifically.** Investigate why graph message passing loses to no-graph and why collusion AUROC is below chance. Use a declared protocol; avoid repeatedly tuning to the same test folds.
4. **Expand baseline/ablation and robustness protocol.** Include the planned MLP/R-GCN, autoencoder, XGBoost/gradient boosting, OpenWGL/graph anomaly baseline, camouflage tier sweeps, alpha/gamma sweeps, shortcut audits, cluster stability and subgroup/fairness reports.
5. **Ground the data schema.** Pin any Synthea release/configuration, map real exported schemas transparently, and replace synthetic Admission/Diagnosis proxies where appropriate. Synthea alone does not provide fraud labels.
6. **HBP provenance.** Manually verify any rates against the official NHA PDF and record source page/version/reviewer. Do not claim official package rates without this verification.
7. **Real claim-level evaluation.** Only with an approved, appropriately labelled claim dataset and review governance. Never present provider-level `PotentialFraud` labels as claim-level fraud-mechanism truth.
8. **External student submission.** Check course portal, required files, AI-assistance disclosure and whether the instructor invitation is still pending. Upload PPTX binaries to GitHub only if needed; sync an external Overleaf copy only if it exists.

## Guardrails for the next AI

- Do not claim “project fully validated” or “real-world fraud detection completed.”
- Do not claim the custom model is canonical PyG HGT/R-GCN.
- Do not claim graph message passing improved results; this paired experiment found the opposite on the current simulator.
- Do not invent clinical ICD codes, NHA rates, patient/claim labels or real-world metrics.
- Keep provider-level public dataset experiments distinct from claim-level synthetic mechanism experiments.
- Keep human review as a required safeguard. The output is for research/audit triage, not automatic adverse claim decisions.
