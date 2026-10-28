# Final Submission Checklist — Insurance Intelligence / OW-ERF

This checklist separates completed repository work from actions requiring external approval, human source review, or real labelled data.

## Built, tested and committed

- [x] Individual project repository: `ishantpant2121/MUJ-DS-2430010379`.
- [x] Policy recommender and Streamlit prototype.
- [x] Tabular OW-ERF and temporal relation-summary pilots.
- [x] Standalone LSTM, 1D-CNN and Transformer benchmark; controlled 60-epoch shared-input comparison.
- [x] Custom typed-node temporal heterogeneous graph baseline with eight typed relation channels, strict-past snapshots and relation attention.
- [x] Graph tests for schema, gradients and temporal leakage; main Python CI passes on Python 3.11 and 3.12.
- [x] Full temporal graph synthetic benchmark: four held-out mechanisms × three generator seeds, 2,400 claims per run, eight-epoch cap (12 runs total).
- [x] Main paper source and compiled PDF include graph results, limitations and the custom-encoder scope note.
- [x] Results summary JSON/CSV and per-mechanism table committed under `results/temporal_heterograph_full/`; full reports, predictions and model weights are retained in GitHub Actions artifact [40562936336](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40562936336).
- [x] Synthea CSV-to-graph adapter and input-hash manifest; it does not invent fraud labels.
- [x] HBP package catalogue template and provenance validator; no official rates are prefilled.
- [x] Updated OW-ERF and PRD editable PowerPoint decks are prepared as submission downloads; they are not committed to GitHub because the available repository operation could not transfer binary PPTX files.
- [x] Both updated decks rendered; automated slide-overflow checks passed.

## Remaining before external submission

- [ ] Verify every cited publication claim-by-claim against the original publication and check external URLs.
- [ ] If a Synthea export is used in a future study, pin its release/commit and exporter configuration and report it separately; Synthea has no fraud-mechanism ground truth by default.
- [ ] If including HBP package rates, transcribe and human-verify each package and source page; do not substitute synthetic placeholders.
- [ ] Complete the broader baseline/ablation plan, shortcut and subgroup audits, and cluster-stability analysis.
- [ ] Obtain appropriately labelled real claim-level fraud data or an approved evaluation design before claiming real-world detection performance.
- [ ] Confirm the course rubric, final submission portal/file formats and AI-assistance disclosure requirements.
- [ ] Accept the instructor's GitHub collaborator invitation if still pending.
- [ ] Synchronize an external Overleaf copy only if one is still in use.
- [ ] Upload the two editable PPTX files under `presentations/` if the instructor requires decks to be present in the repository.

## Measured synthetic graph results

Across the 12 runs, mean ± SD was: known-class accuracy 0.954 ± 0.012, macro-F1 0.645 ± 0.107, novel-mechanism AUROC 0.732 ± 0.170, novel AP 0.137 ± 0.152, novel recall at nominal α=0.05 of 0.180 ± 0.200, known false-rejection rate 0.044 ± 0.009, and escalation rate 0.046 ± 0.009. Held-out collusion AUROC was 0.494 ± 0.049 (near chance); unbundling recall was also weak. See [full result interpretation](TEMPORAL_GRAPH_BENCHMARK_RESULTS.md).

## Limitations that must remain explicit

- All current claim-level mechanism results are from parametric synthetic data, not real insurance claims.
- Provider-level `PotentialFraud` labels are separate from claim-level fraud-mechanism ground truth.
- The graph model is custom PyTorch R-GCN/HGT-style message passing, not canonical PyG `HGTConv`/`RGCNConv`; Admission and Diagnosis are synthetic proxies.
- The explained-rarity gate's novel recall matched raw conformal rejection in the reported graph experiment, so this run does not show additional novel-case recall improvement from that gate.
- The reported zero rare-severe false-rejection rate has an unresolved denominator/reporting limitation and must not be interpreted until the denominator is preserved and audited.
- Alerts are research triage flags for human review, not fraud determinations and not suitable for automated adverse claim decisions.
