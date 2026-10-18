# Reproduce the M2 technical foundation

Use Python **3.12** for the closest match to the recorded environment. Reference runs used 3.12.14 on Linux x86-64 with NumPy 2.3.5, SciPy 1.17.0 and Numba 0.68.0. No GPU, account, API key or proprietary inference library is needed.

Run the following from this `Code/` directory.

## 1. Install

```bash
python -m venv .venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

If activation is restricted, invoke `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux in place of `python`. Setup requires internet access for packages. The full experiment also needs the official 6 MB MovieLens archive, unless supplied locally.

## 2. Check the mathematics and show a quick working pipeline

```bash
python -m matchbox_m2.validation
python -m unittest discover -s tests -v
python -m matchbox_m2.demo
```

The two validation commands expose the same nine checks; either one is sufficient. The default demo uses six **synthetic** ratings and is labelled accordingly. It is a quick execution check, not benchmark evidence. Numba compiles on its first invocation, so that invocation can take longer.

## 3. Reproduce the full benchmark

```bash
python scripts/reproduce.py
```

This sets the numerical thread variables to one, runs three full MovieLens splits and both K=2 and K=8, and generates new evidence under `../Results/reproduced/`. Roughly 300-600 MB of working memory is a useful allowance; EP's K=8 site arrays alone occupy 219.77 MiB. Actual speed depends on hardware. The reference full run took about 110 seconds after dependencies were available; installation/download time is additional.

The script does not overwrite the reference evidence or report. Repeating it requires a new output directory:

```bash
python scripts/reproduce.py --output ../Results/reproduced_second
```

To use an existing official dataset directory containing `ratings.dat`, `users.dat` and `movies.dat`:

```bash
python scripts/reproduce.py --data-root /absolute/path/to/ml-1m --output ../Results/reproduced_local
```

Equivalent low-level experiment command (set thread variables yourself if benchmarking):

```bash
python -m matchbox_m2.experiment --config configs/m2.json --output ../Results/new_run
```

Use a copied config for additional studies. The report describes the supplied reference config; changing it requires revising the interpretation, not only rerunning a command.

## 4. Inspect a saved real model

```bash
python -m matchbox_m2.demo --checkpoint ../Results/reference/matchbox_seed42_k8_ep5.npz --mapping ../Results/reference/id_mapping.npz --user-id 1 --item-id 1
```

The output contains five normalized probabilities and their expected rating. The queried pair is not guaranteed to be held out; benchmark performance comes from the saved test splits, not this demo. Checkpoints contain derived posterior parameters and no raw rating records.

## 5. Generate figures/tables and rebuild the PDF

The source package already includes the generated figures and tables. To regenerate them from the reference metrics:

```bash
python -m matchbox_m2.report_assets
python scripts/build_report.py
```

PDF compilation requires a LaTeX installation with `pdflatex` and `bibtex` (TeX Live or MiKTeX). Alternatively upload **the contents of `Report/latex/`** as an Overleaf project and compile `main.tex` with pdfLaTeX. Its figures are local to that folder.

To inspect an alternative run without changing the report sources:

```bash
python -m matchbox_m2.report_assets --results ../Results/new_run --latex ../Results/new_run/report_assets
```

## Source map

| Module | Responsibility |
|---|---|
| `model.py` | Gaussian natural parameters, stable ordinal moment matching, VMP product messages, ADF/EP sites and prediction |
| `graph.py` | Explicit variables, factors and edges for a small graph instance |
| `data.py` | Official download, SHA-256 checks, ID mapping, splits and audits |
| `baselines.py` | Mean/histogram baselines, biased MF and validation-only noise fitting |
| `metrics.py` | MAE, RMSE, NLL, Brier, ECE, median MAE and intervals |
| `experiment.py` | Paired experiment, seed control, logs, evidence and checkpoint |
| `validation.py` | Independent exact-case checks and algorithmic invariants |
| `validation_study.py` | Two-dimensional quadrature comparison exposing approximation error |
| `report_assets.py` | Figures, tables, paired differences and concrete graph export |
| `demo.py` | Synthetic end-to-end demo or saved-checkpoint query |

## Timing and reproducibility limits

- Fit timings exclude JIT compilation, dataset I/O and KPI evaluation. MF includes validation checkpointing and scale calibration.
- Prediction time covers vectorized probabilities for the whole test batch, not a production serving stack.
- ADF stores no rating sites; EP keeps two natural parameters per touched variable per training rating.
- Exact values are expected to agree closely with the pinned environment. CPU timing and very small floating-point differences can vary.
- No second ADF pass is permitted: repeated data without removing old sites would double-count evidence.
- Core source hashes recorded at run time appear in `Results/reference/source_hashes_at_run.json`. The package manifest covers later-added plotting, documentation and packaging files too.
