# Final Submission Checklist — Insurance Intelligence / OW-ERF

This checklist distinguishes the completed repository artifacts from research and administrative work that must not be falsely marked complete.

## Completed in GitHub main

- [x] Policy recommender prototype and Streamlit application.
- [x] OW-ERF tabular and temporal relation-summary pilots.
- [x] Standalone LSTM/CNN/Transformer models and controlled 60-epoch shared-input comparison.
- [x] Custom typed-node temporal graph encoder, training/evaluation script and strict-past graph tests.
- [x] Full graph synthetic benchmark: 12 runs (four held-out mechanisms × three generator seeds; 2,400 synthetic claims per run; eight-epoch cap).
- [x] Relation-free matched control benchmark: same 12 held-out-mechanism/generator-seed combinations.
- [x] Synthea CSV graph preparation adapter with input-file hash manifest and explicit no-fraud-label warning.
- [x] HBP catalogue CSV template and provenance validator; it does not fabricate or auto-transcribe official rates.
- [x] Python CI passed on Python 3.11 and 3.12.
- [x] Paper source and PDF build successful on the latest CI run.
- [x] Results summaries are tracked under results/temporal_heterograph_full/.
- [x] Full per-fold graph output and matched relation-free control are preserved as GitHub Actions artifacts:
  - Full graph + smoke + no-graph control: https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40680366943
  - Earlier full graph run: https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40562936336
- [x] Repository audit found no zero-byte files, .gitkeep placeholders, or tracked empty-folder placeholders.

## Latest matched synthetic results — full graph vs relation-free control

Both arms were run across 12 folds (four held-out fraud mechanisms × three generator seeds), 2,400 generated claims per fold, CPU and an eight-epoch cap.

| Metric | Full graph mean ± SD | No-graph control mean ± SD |
|---|---:|---:|
| Known-class accuracy | 0.9566 ± 0.0111 | 0.9590 ± 0.0114 |
| Known macro-F1 | 0.6516 ± 0.0946 | 0.6655 ± 0.0998 |
| Novel-mechanism AUROC | 0.6965 ± 0.1662 | 0.7062 ± 0.1895 |
| Novel average precision | 0.0899 ± 0.0687 | 0.0980 ± 0.0741 |
| Novel recall at nominal alpha 0.05 | 0.1077 ± 0.1263 | 0.1272 ± 0.1273 |
| Known false-rejection rate | 0.0391 ± 0.0132 | 0.0380 ± 0.0118 |
| Rare-severe benign false-rejection | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Escalation rate | 0.0401 ± 0.0131 | 0.0396 ± 0.0115 |

These results do **not** show a benefit from graph message passing in this synthetic generator: the no-graph control is slightly better on mean AUROC/AP/novel recall and similar on known-class metrics. This is a negative result, not a reason to claim the graph improves fraud detection. Collusion detection remains near chance in the full-graph runs, and novel recall is low/variable. The severe-benign false-rejection zero is limited to this generator and the documented subset; it is not evidence of real-world safety.

## Remaining before final external submission

- [ ] Claim-by-claim verification of citations and all manuscript factual statements against original publications. Successful BibTeX compilation does not prove citation accuracy.
- [ ] Complete the broader baseline/ablation protocol, shortcut audit, camouflage/alpha sweeps, cluster-stability analysis and subgroup/fairness analysis.
- [ ] If using Synthea in additional work, pin its version/commit and report the export settings. Synthea trajectories do not have fraud-mechanism labels by default, so they cannot alone support claim-level fraud AUROC.
- [ ] Human-check any HBP package/rate transcriptions against the official NHA source PDF and record page-level provenance.
- [ ] Obtain an appropriate, approved labelled claim-level evaluation design before claiming real-world fraud performance.
- [ ] Confirm the course rubric, required submission portal/format and AI-assistance disclosure.
- [ ] Accept the instructor's collaborator invitation if still pending; synchronize external Overleaf only if it is still used.
- [ ] Upload the editable PowerPoint files under presentations/ only if the course requires those binaries in GitHub. There are no presentation PPTX files currently tracked in main.

## Research scope that must remain explicit

- The graph code is a custom PyTorch typed-node temporal R-GCN/HGT-style model, not canonical PyG HGTConv/RGCNConv.
- Admission is represented by one synthetic node per claim and Diagnosis is a severity-bin proxy in this generator.
- Current claim-level mechanisms/results are parametric synthetic data, not real insurance claims.
- Provider-level PotentialFraud labels in the separate public dataset are not claim-level fraud-mechanism ground truth.
- The explained-rarity gate did not increase novel recall in the full graph benchmark; its gate recall equalled raw conformal rejection in all 12 folds.
- All outputs are research/audit-triage flags for human review, not fraud determinations and not suitable for automated adverse claim decisions.

## Useful files

- Main paper source: paper/main.tex
- Model and protocol: docs/TEMPORAL_HETEROGRAPH_MODEL.md
- Graph result notes: docs/TEMPORAL_GRAPH_BENCHMARK_RESULTS.md
- Implementation map: docs/OW_ERF_V02_IMPLEMENTATION_MAP.md
- Training script: scripts/train_temporal_heterograph.py
- Data adapter: scripts/prepare_synthea_graph.py
- HBP provenance validator: scripts/validate_hbp_catalog.py
