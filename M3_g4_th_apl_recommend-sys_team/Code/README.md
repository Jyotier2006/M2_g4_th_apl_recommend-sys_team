# Run and reproduce M3

The active implementation uses procedural functions, NumPy arrays, dictionary
state and explicit loops (CP style). Numerical kernels are accelerated by Numba.
No GPU is required. `src/reference/original_model.py` is the unchanged M2
reference used only to verify equivalence; its class-based API is not the M3 driver.

## Setup

Open a terminal in this `Code` directory. Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows, or
`source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install -r requirements.txt
python tests/check.py
```

The nine checks test original-M2 equivalence, ADF=EP1, site bookkeeping,
truncated Gaussian moments, probability metrics, valid graph precision,
decreasing MAP energy, influence of graph edges, and gamma=0 invariance.

The first invocation compiles kernels. Warm-up compilation is excluded from
reported fitting times. Exact versions used for each recorded run are in its
`environment.json`; broad install ranges are supplied for practical portability.

## Full experiment commands

Each command requires a **new output folder**. Existing runs are never overwritten.
The loader downloads/checks the official MovieLens 1M data if absent. Internet
is needed only for installing packages and fetching the dataset.

```bash
python src/experiments.py --data ../Data/ml-1m --out ../Results/reproduced_main --stage main
python src/experiments.py --data ../Data/ml-1m --out ../Results/reproduced_robustness --config configs/robustness.json --stage robustness
python src/experiments.py --data ../Data/ml-1m --out ../Results/reproduced_sensitivity --stage sensitivity
python src/inference_validation.py --out ../Results/reproduced_validation
```

Main seeds are 42–46. Robustness and sensitivity use 42–44. These are full-data
CPU experiments, not quick synthetic demos; allow tens of minutes or more,
depending on hardware. Keep model runs sequential when comparing wall times.

For a single-seed check:

```bash
python src/experiments.py --data ../Data/ml-1m --out ../Results/check_seed42 --stage main --seeds 42
```

For one isolated condition:

```bash
python src/experiments.py --data ../Data/ml-1m --out ../Results/check_noise --config configs/robustness.json --stage robustness --seeds 42 --conditions noise_0.1
```

`--conditions` filters named experiments; it does not alter their numerical
settings. Available robustness names are `retain_0.25`, `retain_0.5`,
`noise_0.1`, `noise_0.2`, `graph_top_5`, `graph_top_50`, `ratings_only`,
`genres_only`, `cold_items`, and `cold_users`.

The recorded robustness execution was interrupted after seed 44's
`graph_top_5`. Its completed conditions were retained; the five remaining
conditions are in `Results/robustness_remaining`. Analysis merges these without
duplicates. A fresh uninterrupted robustness command produces all conditions
in one directory. Per-run source/configuration records document this recovery.

## Regenerate analysis and report

For the supplied recorded folders:

```bash
python src/analyze.py --results ../Results --report ../Report/latex
```

For fresh results, place the new stage folders under a separate result parent
with names `main`, `robustness`, `sensitivity`, and `validation`, and pass that
parent through `--results`. Do not mix old and new runs by appending CSV rows.
Analysis checks paired user counts for its bootstrap intervals. Figures and
summaries are regenerated from raw CSVs and per-user error aggregates.

The editable report is LaTeX. With a TeX distribution and `latexmk` installed:

```bash
cd ../Report/latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The distributed PDF is already compiled, so TeX is unnecessary for model runs.

## Query a saved checkpoint

```bash
python src/predict.py --model ../Results/main/models/42_GMRF_MAP.npz --mapping ../Results/main/id_mapping.npz --user 1 --movie 1
python src/predict.py --model ../Results/main/models/42_EP5.npz --mapping ../Results/main/id_mapping.npz --user 1 --movie 1
```

This prints five star probabilities, their expected rating, and the most likely
star. A queried pair is not automatically a test observation; the command is a
prediction demonstration. IDs outside the checkpoint vocabulary are rejected.

## Experimental decisions

- Main MAP grid: gamma 0, 1, 10, 100; same 25-epoch budget and initialization.
- MAP checkpoint and gamma: validation clipped RMSE; probability scale: validation NLL.
- EP noise grid: variances 0.5, 1, 2; selection: validation NLL.
- Robustness: fixed gamma 10 versus 0, selected from clean-validation evidence;
  identical conditions within each pair. Graphs are rebuilt from condition training data.
- Common primary decoder: expected star rating. Clipped MAP/SGD scores are secondary.
- No test metric selects a model. All fixed comparisons and limitations are disclosed.
- Mean ± sample SD describes split variation; paired user bootstrap gives
  conditional intervals, not independent-seed or universal significance claims.

## Source map

| File | Purpose |
|---|---|
| src/bayes.py | Original hybrid EP/VMP kernels; optional noise/prior/damping settings |
| src/gmrf.py | Genre mapping, graph construction, proper GMRF penalty, alternating MAP |
| src/sgd_baseline.py | Unchanged original M2 SGD numerical kernel |
| src/experiments.py | Data splits, comparisons, validation selection, robustness and exports |
| src/metrics.py | RMSE, MAE, NLL, Brier, ECE and rating intervals |
| src/inference_validation.py | Exact Gaussian / Gauss–Seidel / Gibbs diagnostic and quadrature |
| src/analyze.py | Tables, figures, user-cluster bootstrap and error analysis |
| src/predict.py | Saved-model probability query |
| src/data.py | Official dataset downloader, hashes, mapping and split audit |
