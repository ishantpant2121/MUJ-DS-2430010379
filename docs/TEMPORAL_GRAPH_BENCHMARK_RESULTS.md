# Temporal graph benchmark — measured results

The first complete temporal heterogeneous graph benchmark ran successfully in GitHub Actions on CPU: **12 runs** (four held-out mechanisms × seeds 42, 43, 44), 2,400 synthetic claims per run, an eight-epoch cap, batch size 64. Workflow: https://github.com/ishantpant2121/MUJ-DS-2430010379/actions/runs/40562936336. The full artifact `temporal-heterograph-full-40562936336` contains per-fold reports, test predictions and checkpoints and is retained for 90 days from the run.

## Aggregate results (mean ± standard deviation across 12 folds)

| Metric | Mean ± SD |
|---|---:|
| Known-class accuracy | 0.954 ± 0.012 |
| Known macro-F1 | 0.645 ± 0.107 |
| Novel AUROC | 0.732 ± 0.170 |
| Novel average precision | 0.137 ± 0.152 |
| Novel recall at α = 0.05 | 0.180 ± 0.200 |
| Known false-rejection rate | 0.044 ± 0.009 |
| Escalation rate | 0.046 ± 0.009 |

## Held-out mechanism results (mean ± SD across three seeds)

| Held-out mechanism | Novel AUROC | Novel AP | Novel recall at α = 0.05 |
|---|---:|---:|---:|
| Upcoding | 0.835 ± 0.070 | 0.162 ± 0.033 | 0.260 ± 0.065 |
| Unbundling | 0.714 ± 0.111 | 0.045 ± 0.017 | 0.069 ± 0.064 |
| Phantom billing | 0.886 ± 0.052 | 0.314 ± 0.217 | 0.364 ± 0.310 |
| Collusion | 0.494 ± 0.049 | 0.027 ± 0.009 | 0.026 ± 0.044 |

## Interpretation and limitations

- The results are parametric synthetic data, not real insurance claims.
- Collusion detection is near chance; unbundling recall is also weak. High known-class accuracy should not be mistaken for reliable novel-mechanism discovery.
- The explained-rarity gate's novel recall exactly equaled raw conformal novel recall in all 12 runs. In this experiment it did not further screen the novel cases already rejected by the conformal rule.
- The aggregate report did not preserve the denominator for the rare-severe-benign subset; its reported zero false-rejection rate should not be interpreted until the code records and reports those counts.
- The custom PyTorch encoder is an R-GCN/HGT-style baseline, not canonical PyG HGTConv/RGCNConv. Admission is a one-per-claim proxy; Diagnosis is a severity-bin proxy.
- Broader baseline/ablation comparisons, shortcut and subgroup audits, verified Synthea/HBP grounding and evaluation on appropriately labelled real claims remain outstanding.
