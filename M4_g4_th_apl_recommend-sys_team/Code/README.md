# Run and reproduce the final project

The active implementation uses procedural functions, NumPy arrays, dictionary
state and explicit loops (CP style). Numerical kernels are accelerated by Numba.
No GPU is required. `src/reference/original_model.py` is the unchanged M2
reference used only to verify equivalence; the final runner uses procedural functions.

## Setup

Open a terminal in this `Code` directory. Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows, or
`source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install -r requirements-lock.txt
python tests/check.py
```

The nine checks test original-M2 equivalence, ADF=EP1, site bookkeeping,
truncated Gaussian moments, probability metrics, valid graph precision,
decreasing MAP energy, influence of graph edges, and gamma=0 invariance.

The first invocation compiles kernels. Warm-up compilation is excluded from
reported fitting times. Exact versions used for each recorded run are in its
`environment.json`; the lock file records the verified replay environment. If a pinned wheel is
unavailable on your platform, use the broad ranges in `requirements.txt`.

## Full experiment commands

For comparable CPU timing, use one thread and run jobs sequentially. On
Linux/macOS set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` in your shell before running the scripts. The exact
settings and run completion are recorded in each output folder.

Each command requires a **new output folder**. Existing runs are never overwritten.
The loader downloads/checks the official MovieLens 1M data if absent. Internet
is needed only for installing packages and fetching the dataset.

```bash
python src/experiments.py --data ../Data/ml-1m --out ../Results/reproduced/main --stage main
python src/experiments.py --data ../Data/ml-1m --out ../Results/reproduced/robustness --config configs/robustness.json --stage robustness
python src/experiments.py --data ../Data/ml-1m --out ../Results/reproduced/sensitivity --stage sensitivity
python src/inference_validation.py --out ../Results/reproduced/validation
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

These commands use the recorded outputs and require no dataset download:

```bash
python tests/check_results.py --results ../Results
python src/analyze.py --results ../Results --report ../Results/regenerated_report
python src/final_kpis.py --results ../Results --out ../Results/final
```

`analyze.py` writes the CSVs and plots into `Results/analysis/`. Its generated
LaTeX tables land in `Results/regenerated_report/generated/`; copy the tables
with matching names to `Report/LaTeX/tables/` if rebuilding from new data.
Copy the corresponding plots to `Report/LaTeX/figures/`. The final KPI script
writes `quality.tex`, `cost.tex`, and `inference_tradeoff.pdf` under
`Results/final/`; these feed the like-named report assets.
For new runs pass `--results ../Results/reproduced` instead, so the recorded
CSVs stay unchanged.

Compare a fresh seed-42 run to the archived run:

```bash
python src/final_kpis.py --results ../Results --out ../Results/final --rerun ../Results/check_seed42
```

This compares the recorded split hashes and selected scores and settings;
hardware-dependent timings are excluded. The precomputed benchmark can also
be explored in `Final_Results.ipynb` with Jupyter, started in `Code/`.
To compile the report, see [Report/README.md](../Report/README.md).

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
- No test metric selects a model.
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
| src/final_kpis.py | Five-split tables, inference timing plot, replay comparison |
| src/predict.py | Saved-model probability query |
| src/data.py | Official dataset downloader, hashes, mapping and split audit |
