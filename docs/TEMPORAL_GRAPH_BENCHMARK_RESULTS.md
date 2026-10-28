# Temporal Graph Benchmark Results — Full graph vs relation-free ablation

## Reproducibility
Both arms use the parametric synthetic claim generator, four held-out mechanisms (upcoding, unbundling, phantom billing, collusion), seeds 42/43/44, 2,400 claims per fold, an eight-epoch cap, and CPU. Each arm contains 12 fits. The no-graph arm retains the same model and downstream classifier/calibration/gating path but zeros entity-message contributions.

Successful combined workflow and artifacts: https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40680366943
- Full graph artifact: temporal-heterograph-full-40680366943
- Relation-free artifact: relation-free-ablation-40680366943

## Matched aggregates
All values are means ± sample standard deviation across 12 runs.

| Metric | Full graph | No-graph control |
|---|---:|---:|
| Known-class accuracy | 0.9566 ± 0.0111 | 0.9590 ± 0.0114 |
| Known macro-F1 | 0.6516 ± 0.0946 | 0.6655 ± 0.0998 |
| Novel-mechanism AUROC | 0.6965 ± 0.1662 | 0.7062 ± 0.1895 |
| Novel average precision | 0.0899 ± 0.0687 | 0.0980 ± 0.0741 |
| Novel recall at alpha=0.05 | 0.1077 ± 0.1263 | 0.1272 ± 0.1273 |
| Known false-rejection rate | 0.0391 ± 0.0132 | 0.0380 ± 0.0118 |
| Rare-severe benign false-rejection | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Gated novel recall | 0.1077 ± 0.1263 | 0.1212 ± 0.1245 |
| Escalation rate | 0.0401 ± 0.0131 | 0.0396 ± 0.0115 |

These are descriptive synthetic outcomes, not estimates of real-world performance. The matched no-graph control is slightly ahead on mean AUROC, AP, novel recall, known accuracy, and macro-F1. This run does not demonstrate a benefit from relational message passing.

## Full graph results by held-out mechanism

| Mechanism | Novel AUROC | Novel AP | Novel recall at alpha=0.05 |
|---|---:|---:|---:|
| Upcoding | 0.7510 ± 0.0164 | 0.1307 ± 0.0705 | 0.1505 ± 0.0655 |
| Unbundling | 0.6876 ± 0.0221 | 0.0359 ± 0.0081 | 0.0000 ± 0.0000 |
| Phantom billing | 0.8920 ± 0.0192 | 0.1613 ± 0.0304 | 0.2545 ± 0.1376 |
| Collusion | 0.4553 ± 0.0404 | 0.0314 ± 0.0187 | 0.0256 ± 0.0444 |

The no-graph control’s corresponding AUROCs were 0.7892 ± 0.0455 (upcoding), 0.7058 ± 0.0257 (unbundling), 0.9071 ± 0.0209 (phantom billing) and 0.4226 ± 0.0532 (collusion). This is a particularly important result: relational context did not improve these generated data’s collusion detection and the full graph arm was near/below chance.

The explained-rarity gate’s novel recall equals raw conformal rejection recall in the full-graph arm across all 12 folds. The gate therefore did not add recall in this benchmark. The reported zero rare-severe benign false-rejection is limited to this generator and does not imply real-world safety.

## Scope limits and next research
- These are parametrically generated claims; no real insurer claim labels are used.
- The custom encoder is R-GCN/HGT-style PyTorch message passing, not canonical PyG HGTConv/RGCNConv.
- Admission is a one-per-claim synthetic proxy and Diagnosis a severity-bin proxy.
- Next: investigate near-chance collusion, revise the generator/shortcut design with a pre-declared protocol, run broader baselines and ablations, perform subgroup/cluster audits, verify source provenance and evaluate on appropriately labelled claims where approved.
