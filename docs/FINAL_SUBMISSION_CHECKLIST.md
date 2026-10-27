# Final Submission Checklist — Insurance Intelligence / OW-ERF

This checklist separates artifacts built in the repository from actions that require external approval or data.

## Current code and document artifacts

- [x] Individual project repository retained as ishantpant2121/MUJ-DS-2430010379.
- [x] Policy recommender and Streamlit application.
- [x] Tabular OW-ERF prototype and synthetic temporal relation-summary pilot.
- [x] Standalone LSTM, continuous-input 1D-CNN and Transformer benchmark.
- [x] Controlled 60-epoch shared-input deep-model comparison.
- [x] Explicit typed-node temporal heterogeneous graph encoder and training/evaluation script on the research branch.
- [x] Temporal leakage, schema, gradients and future-feature invariance tests added.
- [x] Reproducible CI smoke workflow plus manually selectable full 12-run benchmark workflow.
- [x] Main paper PDF from the previously evaluated experiments.
- [x] Editable OW-ERF combined presentation and Insurance Intelligence PRD presentation are available as separate session downloads; upload their binary PPTX files to presentations/ if the instructor requires them in GitHub.

## Before submission

- [ ] Review the temporal heterogeneous graph pull request and confirm CI has passed; merge it only after checks pass.
- [ ] Run or download the full graph benchmark artifact and update the paper with its actual metrics. Do not paste smoke metrics into the paper as if they were final study results.
- [ ] Verify cited publications claim-by-claim against source publications and confirm all external URLs.
- [ ] If required, transcribe a small HBP package/rate subset from the official NHA document, recording source page, version, review date and reviewer; do not represent synthetic placeholder rates as official rates.
- [ ] If a Synthea release is used, pin its release/commit and exporter configuration. Treat Synthea as synthetic clinical data, not fraud-labeled ground truth.
- [ ] Confirm course rubric, submission file formats/URLs, and AI-assistance disclosure.
- [ ] Accept the instructor's GitHub collaborator invitation if it is still pending.
- [ ] Synchronize an external Overleaf copy only if one is still being used.
- [ ] Upload the final PPTX files to the repository if editable decks are required.
- [ ] Make final PDF/deck exports from the same final source revision; verify the rendered PDF and all slide layouts after the last content changes.

## Research claims and limitations that must remain explicit

- The existing tabular, relation-summary and deep-model result tables are based on parametric synthetic claim data.
- The provider-level PotentialFraud task is separate from claim-level fraud-mechanism evaluation.
- Synthea provides synthetic health-care trajectories but does not label insurance fraud mechanisms. It cannot by itself establish claim-level fraud detection performance.
- The graph module's current admission and diagnosis nodes are proxies, and the model is a dependency-light R-GCN/HGT-style implementation rather than canonical PyG HGTConv/RGCNConv.
- The full graph benchmark is only considered empirically complete once its workflow produces an artifact from the complete 4-fold × 3-seed protocol and the metrics are reviewed.
- No model is ready to make automated adverse claim decisions; it remains a research/audit-triage prototype requiring human review and external validation.
