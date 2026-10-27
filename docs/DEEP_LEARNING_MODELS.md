# Deep-learning model guide — OW-ERF / Insurance Intelligence

## What the three models do

The design uses three complementary input views. They are not interchangeable: each model can only learn signal present in its input modality.

### 1. LSTM — recent claim-history sequence
- **Input in the standalone baseline:** a 10-step sequence with six numeric channels.
- **Layer design:** two-layer LSTM, hidden size 48, layer normalisation, dropout and a linear known-class classifier.
- **Research intent:** learn the provider's recent rhythm—claim amount, length of stay, inter-claim gaps and similar history statistics—and recognise bursts or deviations from recent patterns.
- **Training objective:** class-weighted cross-entropy on known classes; AdamW optimizer; checkpoint selected by known-class validation cross-entropy.
- **Current limitation:** generator histories are synthetic proxies. The LSTM benchmark does not ingest real hospital claim streams or a complete graph.

### 2. 1D-CNN — ordered procedure/line-item sequence
- **Input in the standalone baseline:** a 14-token ordered procedure/line-item sequence; zero is reserved for padding.
- **Layer design:** learned token embeddings, two Conv1D blocks (kernel widths 3 and 5), batch normalisation, dropout, global mean/max pooling, projection and a known-class classifier.
- **Research intent:** detect local billing-code patterns, such as overlapping lines or a separately charged service that may already be included in a package.
- **Training objective:** class-weighted cross-entropy with AdamW; checkpoint selected on validation cross-entropy.
- **Current limitation:** vocabulary and unbundling patterns use synthetic codes, not an authoritative Indian coding/rate catalogue.

### 3. Transformer — relational-context tokens
- **Input in the standalone baseline:** six tokens of 16 dimensions representing claim-related context; the input generator supplies a padding mask.
- **Layer design:** input projection to hidden size 48, a learnable CLS token, two TransformerEncoder layers with four attention heads, GELU feed-forward blocks and a known-class classifier.
- **Research intent:** self-attention weights relationship context, e.g. recurring doctor/hospital/package/diagnosis combinations; CLS collects a claim-level embedding.
- **Training objective:** class-weighted cross-entropy with AdamW; checkpoint selected on validation cross-entropy.
- **Current limitation:** tokens are synthetic proxies; the standalone Transformer does not implement arbitrary-node temporal HGT/R-GCN message passing.

## How training and testing are organised

The script scripts/train_evaluate_deep_models.py trains each architecture independently using the same generated data seeds and held-out-mechanism test set:

1. Generate 2,400 records for each generator seed (default seeds 42, 43 and 44).
2. Split labels into train/calibration/test partitions, then carve a known-only validation partition out of the train pool.
3. Hold one fraud mechanism out of the model's train, validation and calibration data; the test set includes known and held-out mechanism claims.
4. Fit feature scaling on training rows only. LSTM uses the history sequence; CNN uses integer code sequences; Transformer uses context tokens with the padding mask.
5. Train for up to five epochs by default and keep the epoch with the lowest validation cross-entropy.
6. Fit energy and embedding Mahalanobis reference scores with validation known-class data; compute split-conformal p-values using a separate known-only calibration partition at alpha = 0.05.
7. Evaluate the frozen model on the held-out test partition.

The 3 models × 4 held-out mechanisms × 3 generator seeds create 36 trained model fits. Checkpoints and per-run metric records are keyed by model, withheld mechanism and seed.

## Controlled architecture comparison: native vs shared context

The native benchmark gives each architecture its intended input modality, so performance differences can reflect both architecture and input representation. The controlled ablation supplies the same standardized relational-context tensor (N, 6, 16) and padding mask to the LSTM, continuous-input 1D-CNN, and Transformer. Scaling is fitted on valid training tokens only, padded positions are set to zero after scaling, the LSTM packs valid sequence lengths, the CNN uses mask-aware pooling, and the Transformer uses its key-padding mask. Checkpoint selection uses validation loss; the known-only calibration split and final test split remain separate.

The completed 60-epoch run used 3 generator seeds × 4 held-out mechanisms (upcoding, unbundling, phantom billing, and collusion), with 36 fits in native mode and 36 in shared-context mode. Mean ± SD over 12 held-out mechanism/seed runs per architecture:

| Mode / model | Known accuracy | Known macro-F1 | Novel AUROC | Novel AP | Novel recall at α=0.05 | Known false-rejection |
|---|---:|---:|---:|---:|---:|---:|
| Native LSTM | 0.519 ± 0.098 | 0.376 ± 0.123 | 0.539 ± 0.130 | 0.061 ± 0.073 | 0.057 ± 0.108 | 0.051 ± 0.010 |
| Shared LSTM | 0.979 ± 0.010 | 0.908 ± 0.054 | 0.721 ± 0.210 | 0.169 ± 0.105 | 0.250 ± 0.166 | 0.048 ± 0.019 |
| Native CNN | 0.896 ± 0.016 | 0.379 ± 0.085 | 0.546 ± 0.136 | 0.055 ± 0.032 | 0.078 ± 0.100 | 0.053 ± 0.016 |
| Shared CNN | 0.973 ± 0.016 | 0.865 ± 0.091 | 0.759 ± 0.171 | 0.188 ± 0.141 | 0.354 ± 0.325 | 0.054 ± 0.015 |
| Native Transformer | 0.983 ± 0.009 | 0.926 ± 0.042 | 0.816 ± 0.090 | 0.174 ± 0.103 | 0.349 ± 0.228 | 0.061 ± 0.021 |
| Shared Transformer | 0.983 ± 0.009 | 0.926 ± 0.042 | 0.816 ± 0.090 | 0.174 ± 0.103 | 0.349 ± 0.228 | 0.061 ± 0.021 |

The original native performance gap is substantially affected by modality. The Transformer is unchanged because its native input was already the shared-context tensor. This controlled experiment does not prove a statistically decisive model ranking: there are three generator seeds, novel recall has high variability, and the Transformer known false-rejection rate is slightly above the nominal 5%. Three shared-context LSTM fits and one CNN fit selected epoch 60, so additional convergence/early-stopping analysis remains useful. The experiment is synthetic-only.

The earlier 25-epoch controlled report is preserved at results/controlled_deep_model_comparison/controlled_comparison_summary.json; the current report is results/controlled_deep_model_comparison/controlled_comparison_summary_60ep.json, with per-mode JSON/CSV and selected checkpoints in their respective result directories.

## Metrics and exact interpretation

- **Known-class accuracy:** multiclass accuracy among test examples from classes seen during training; the withheld mechanism is excluded from this accuracy.
- **Known-class macro-F1:** unweighted mean F1 across the known classes; useful where class sizes differ.
- **Known-class balanced accuracy:** average class recall across the known classes.
- **Novel-mechanism AUROC:** ability of the fused novelty score to rank held-out-mechanism examples above all non-held-out test examples.
- **Novel-mechanism average precision:** precision-recall summary for the held-out mechanism as the novelty-positive class.
- **FPR@95% novel recall:** fraction of non-held-out claims whose novelty score is at/above the threshold that attains at least 95% held-out recall.
- **Novel recall at conformal alpha 0.05:** fraction of held-out-mechanism test claims rejected by the calibration-derived p-value rule.
- **Known false-rejection rate:** fraction of known-class test claims rejected by the same rule.
- **Rare-severe benign false-rejection rate:** fraction of generated rare-severe legitimate test claims rejected by the novelty rule.
- **Precision@K:** held-out-mechanism share among the highest novelty-scored K test examples.

These measures answer different questions. A model can be accurate on known classes but weak at recognising unseen mechanisms. A 5% conformal level is calibrated on known-class calibration data; it does not guarantee 5% rejection of novel claims. Results are conditional on the synthetic generator.

## Reproducing the run

~~~bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/train_evaluate_deep_models.py --n-samples 2400 --epochs 5 --batch-size 128 --seeds 42,43,44 --hidden all --output-dir results/standalone_deep_models
python -m pytest tests/test_standalone_deep_models.py -v
~~~

Output files:
- results/standalone_deep_models/standalone_deep_model_metrics.json
- results/standalone_deep_models/standalone_deep_model_metrics.csv
- results/standalone_deep_models/weights/ (one saved PyTorch checkpoint per model/fold/seed)

The aggregate table is documented in the root README and the interactive app can display recorded metrics after the workflow publishes them. Do not present these numbers as real-claim results.
