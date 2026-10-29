# Temporal Graph Benchmark Results

**Status:** matched synthetic graph/control benchmark completed. The full graph and relation-free control were rerun with arm labels checked by CI. Workflow: https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40870549723

## Protocol

- 4 held-out mechanisms: upcoding, unbundling, phantom billing, collusion.
- 3 generator seeds: 42, 43, 44.
- 12 folds per arm; 2,400 synthetic claims per fold.
- CPU, epoch cap 8, batch size 64.
- Chronological train/selection/calibration/deployment split and known-only calibration.
- Both arm summaries record the same source commit, `9680554c611be0e2ee6c60b0038a74fdafdd640b`, with explicit `ablation=full` or `ablation=no_graph`.

## Results (mean ± sample standard deviation across 12 folds)

| Metric | Full graph | Relation-free control |
|---|---:|---:|
| Known-class accuracy | 0.9566 ± 0.0111 | 0.9590 ± 0.0114 |
| Known macro-F1 | 0.6516 ± 0.0946 | 0.6655 ± 0.0998 |
| Novel-mechanism AUROC | 0.6965 ± 0.1662 | 0.7062 ± 0.1895 |
| Novel average precision | 0.0899 ± 0.0687 | 0.0980 ± 0.0741 |
| Novel recall at nominal alpha 0.05 | 0.1077 ± 0.1263 | 0.1272 ± 0.1273 |
| Known false-rejection rate | 0.0391 ± 0.0132 | 0.0380 ± 0.0118 |
| Rare-severe benign false-rejection | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Explained-rarity escalation rate | 0.0401 ± 0.0131 | 0.0396 ± 0.0115 |

## Interpretation

The relation-free control is slightly better on mean AUROC, average precision, novel recall, and known-class metrics. This is a negative result for graph message passing in the current synthetic generator; it does not prove graph structure is useless in real claims. Novel recall is low and variable. Collusion AUROC is near/below chance. The explained-rarity gate did not increase novel recall in the graph arm. The zero rare-severe benign false-rejection rate is specific to the generated cohort and is not evidence of real-world safety.

## Artifacts

- [Full graph + relation-free control workflow](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40870549723)
- [Earlier paper build and tests](https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40692081902)
- [Current manuscript PDF](https://github.com/ishantpant2121/MUJ-DS-2430010379/blob/main/paper/OWERF_Research_Paper.pdf)

The downloadable workflow artifacts preserve per-fold reports, predictions, and model checkpoints. Results remain synthetic-only and do not constitute real claim-level fraud validation.
