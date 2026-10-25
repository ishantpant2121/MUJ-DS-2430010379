# OW-ERF: Open-World Explained-Rarity Filtering for Health-Insurance Claim Fraud

Research prototype accompanying the paper in `paper/`. It evaluates, on **synthetic data**, a pipeline that
escalates a claim only if it is both unfamiliar to the model and unexplained by its clinical context.

**Status:** preliminary. `owerf_core.py` is a tabular prototype (scikit-learn, no deep learning, no graph).
`owerf_deep_encoders.py` (LSTM, 1D-CNN, Transformer branches) is written and unit-tested for shapes and gradients,
but **not trained or evaluated**. No real patient data are used. Results depend on the synthetic generator.

## Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate      Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-deep.txt   # optional, for the deep encoders
```

## Run the experiment
```bash
python owerf_core.py > results/results.txt
```
Leave-one-mechanism-out, 5 seeds x 4 mechanisms. Prints mean +/- std per held-out mechanism.

## Test
```bash
pytest -v
```
`tests/test_core.py` checks the generator, energy score, empirical CDF, conformal p-values (rejection rate near 5%),
and that escalations are a subset of rejections. `tests/test_deep.py` is skipped automatically if PyTorch is missing.

## Layout
```
owerf_core.py             tabular prototype and experiment
owerf_deep_encoders.py    PyTorch encoders and losses (untrained)
tests/                    pytest suite
results/results.txt       output of the run reported in the paper
paper/                    LaTeX source (IEEEtran)
```

## Limitations
Synthetic-only evidence, one baseline, no graph encoder, conformal guarantee is marginal and assumes exchangeability.
See Section VII of the paper.

## AI assistance disclosure
Code and text were drafted with assistance from Claude (Anthropic); the authors reviewed them and are responsible for the content.
