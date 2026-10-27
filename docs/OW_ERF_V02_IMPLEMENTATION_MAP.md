# OW-ERF Manuscript v0.2: implementation map

## Source document and scope

The uploaded **Explain Before You Flag — Manuscript v0.2** is explicitly a pre-implementation registered-report-style design. It intentionally contains no empirical results. This map distinguishes the design described there from the current executable pilot. The uploaded document has not been copied into the GitHub manuscript as if it were an experiment report.

## Eight-stage method comparison

| Manuscript stage | Current prototype implementation | Status / limitation |
|---|---|---|
| 1. Causal temporal heterogeneous claim graph encoder | **owerf_v02_benchmark.py** generates timestamped claim-to-entity edges for provider, hospital, doctor, beneficiary, package and referral-target relations. Its encoder uses relation-specific attention over summaries of earlier events only. | **Partial implementation.** It is causal and relational at the summary level, but not a full arbitrary-node PyTorch-Geometric HGT/R-GCN with explicit typed-node embeddings and neighbor sampling. |
| 2. Episodic pseudo-novel energy-margin training | The episodic loss masks one known fraud class per mini-batch episode; known claims receive cross-entropy and energy-margin loss. The clinical expectation head is trained with Gaussian NLL on legitimate training claims. | Implemented in the pilot; needs ablation and repeated-seed evaluation. |
| 3. Energy + Mahalanobis novelty score | Energy and nearest-class Mahalanobis distances are mapped through validation empirical CDFs and averaged. | Implemented in the pilot. |
| 4. Split-conformal rejection | Uses a chronologically separate known-class calibration window; p = (1 + count(calibration_score >= test_score)) / (n_cal + 1). | Implemented with the stated exchangeability/drift caveat; realized false-rejection rate is reported. |
| 5. Explained-rarity filter | A heteroscedastic target head predicts log amount, bed tier and log length of stay. Thresholds are derived from legitimate calibration claims. A separate structural Mahalanobis reference uses legitimate training context. | Implemented as a pilot. Exogenous-input assumptions and the structural feature definition still need scientific review. |
| 6. Suspicious pool | Escalates only where conformal rejection is true and max(clinical/tau_clinical, structural/tau_structural) > 1. | Implemented in the pilot. This is an audit-triage flag, not a fraud determination. |
| 7. Density-based typology grouping | Uses HDBSCAN when installed (the dedicated workflow installs it); falls back to DBSCAN only if HDBSCAN is unavailable. | Implemented as a grouping baseline; cluster stability and hyperparameter selection are not yet studied. |
| 8. Cluster cards for investigators | Emits cluster size, time span, provider concentration, mean clinical/structural scores, priority and most-shifted observable features. | Implemented as a basic machine-readable cluster card; no production review UI exists. |

## Benchmark and evaluation pieces

Implemented in the current pilot:

- Four parameterised mechanisms: upcoding, unbundling, phantom billing and collusive referral patterns.
- Camouflage tiers 1, 2 and 3 are selectable at runtime.
- S1-style benign clinical-outlier cohorts are reserved for the deployment window and withheld from train/selection/calibration.
- Chronological train, selection, calibration and deployment/test windows.
- The held-out mechanism is absent from train/selection/calibration; labels are stored as separate evaluation metadata and are not model input features.
- Audit metrics include AUROC, average precision, FPR at 95% novel recall, realised known-class false rejection, novel recall after filtering, clinical false-alarm rate before/after the filter, Precision@K, time-to-detection, NMI/ARI, noise fraction, cluster cards and a new-vs-established-hospital burden-fairness proxy.
- Documentation-incompleteness sweep at gamma = 0, 0.1, 0.25 and 0.5 is evaluated without retraining or recalibrating thresholds.
- Baselines implemented in this pilot: Random Forest (one minus maximum known-class probability) and Isolation Forest trained on legitimate training claims.
- The separate **owerf_deep_experiment.py** remains an LSTM/CNN/Transformer synthetic prototype; it is not the same as the causal relational encoder pilot.

## Separately trained LSTM/CNN/Transformer baselines

The three standalone encoders are implemented in `scripts/train_evaluate_deep_models.py` and each is trained as its own known-class classifier under the same leave-one-mechanism-out data splits:

- **LSTM:** two-layer recurrent encoder over the ordered 10-step, six-channel history sequence.
- **1D-CNN:** token embedding and two convolution blocks over 14 ordered procedure/line-item tokens.
- **Transformer:** two-layer Transformer encoder with a CLS token and padding mask over six relational-context tokens.

A disjoint known-only validation partition chooses the checkpoint; a separate known-only calibration partition fits the energy/Mahalanobis novelty transform and split-conformal p-values; the untouched test partition reports known-class accuracy/macro-F1 and held-out-mechanism novelty metrics. The dedicated workflow runs 3 generator seeds × 4 withheld mechanisms × 3 models = 36 fits. Result and weight paths are listed in the root README and the report records the metric semantics. Different input modalities mean this is a comparison of architecture-plus-modality pipelines, not a controlled comparison where all models consume identical tensors.

## Controlled architecture ablation

In addition to the native-modality 36-fit experiment, a controlled ablation runs 36 native-mode fits and 36 shared-context fits (3 seeds × 4 held-out mechanisms × 3 models per mode) at the same 25-epoch maximum. In shared-context mode, all three classifiers consume the same standardized six-token, 16-channel context array and padding mask; training-only scaling, checkpoint selection, known-only calibration and test handling remain unchanged. The 1D-CNN uses a continuous-input convolutional stem only for this ablation; the native CNN keeps its procedure-code embedding. See `results/controlled_deep_model_comparison/controlled_comparison_summary.json` after the controlled workflow publishes its measured report.

## Explicit gaps against the manuscript's full protocol

The following are **not yet complete** and must not be claimed as done:

1. Run Synthea at a pinned commit, ingest its exported clinical trajectories, and document the exact export/configuration used.
2. Curate/version the condition-to-ICD-10 mapping and a 40–60-package catalogue transcribed from the official NHA HBP 2.2 package/rate annex. The current simulator has parametric package rates only and does **not** claim to use official HBP rates.
3. Implement the full graph with typed claim/beneficiary/hospital/doctor/admission/package/diagnosis nodes, typed edges and an arbitrary-node PyG HGT or R-GCN, with neighbor sampling and explicit temporal edge filters.
4. Add all manuscript baselines B1–B8: XGBoost/gradient boosting, autoencoder, HAN, OpenWGL or documented reimplementation, OW-base without pseudo-novel training, and both MLP/R-GCN ablations. This pilot currently has Random Forest and Isolation Forest comparisons.
5. Run the pre-registered 3 generator seeds × 3 model seeds, primary D and A folds, leave-two-out fold(s), all camouflage-tier combinations, the declared ablations, gamma and alpha sweeps, and the premise/shortcut audit.
6. Verify that provider-structure features do not make the synthetic collusion fold trivially easy; keep a deviation log for all pilot changes.
7. Validate cluster stability and reporting over repeated runs before treating clusters as typologies.

Official design references:
- [Synthea synthetic patient generator](https://github.com/synthetichealth/synthea)
- [National Health Authority AB-PMJAY HBP 2.2 manual](https://nha.gov.in/img/resources/HBP-2.2-manual.pdf)

## Running and artifacts

~~~bash
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install hdbscan
python owerf_v02_benchmark.py --n-samples 2400 --epochs 2 --batch-size 128 --hidden all --tier 2 --seed 42 --output-dir results/owerf_v02_pilot
python -m pytest tests/test_owerf_v02_benchmark.py -v
~~~

The report is **results/owerf_v02_pilot/owerf_v02_synthetic_benchmark.json**. Each fold also writes the fold report, test predictions, generated timestamped graph edges and a separate hidden ground-truth table for evaluation. The CI workflow uploads the complete fold artefacts and commits only the aggregate report to **main**.

All outputs from this module are **synthetic pilot measurements**, not results from the uploaded registered-report manuscript, not a real insurance claims dataset, and not a legal finding of fraud.
