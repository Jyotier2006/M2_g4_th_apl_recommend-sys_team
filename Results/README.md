# Initial M2 evidence

`reference/` contains actual output from the submitted CPU implementation on **17 October 2026**, not numbers copied from a paper. There are three full-data seeds, K=2/K=8, independent ADF paths, EP checkpoints 1/3/5 and the M1 baselines.

| File | Meaning |
|---|---|
| `metrics.csv` | Every reported model/seed checkpoint, test KPIs, validation summaries and timings |
| `summary.csv` | Grouped mean and sample SD (`ddof=1`) |
| `learning_trace.csv` | All five EP passes and all 25 MF epochs; validation progress and diagnostics |
| `calibration_bins.csv` | Top-class probability/accuracy bins for every reported run |
| `paired_differences.csv` | Per-seed differences for EP5 versus ADF/MF; descriptive, not significance tests |
| `dataset_audit.json` | Counts, rating histogram, duplicates, density and input hashes |
| `split_audits.json` | Split sizes/hashes and incidental cold-item counts |
| `split_seed_42.npz`, `43`, `44` | Exact train/validation/test row indices into the official data |
| `config.json` | Frozen experiment settings |
| `environment.json` | Python/library versions, platform, command, thread settings and actual UTC timestamps |
| `source_hashes_at_run.json` | Hashes of numerical and experiment sources present when the run began |
| `run.log` | Actual progress and KPI output, including full-data ADF/EP1 equality checks |
| `validation.json` | Nine mathematical/implementation checks, all passing |
| `bilinear_validation.csv` | Independent quadrature versus hybrid posterior moments |
| `bilinear_validation_metadata.json` | Quadrature-stability and local convergence information |
| `graph_instance.json` | Explicit tiny graph with variables, factors and edges |
| `matchbox_seed42_k8_ep5.npz` | Derived posterior parameters for a real reference checkpoint |
| `id_mapping.npz` | Original ID vocabulary corresponding to checkpoint array indices |
| `plots/` | Vector PDFs and preview PNGs generated directly from saved result tables |

## Reading the metrics correctly

`mae` and `rmse` use the expected rating under the predicted categorical distribution. `median_mae` uses the categorical median instead. For MF, `raw_point_mae` and `raw_point_rmse` give the standard clipped continuous prediction; these are deliberately exposed so the probability wrapper cannot hide a stronger conventional baseline.

NLL is in natural-log units per rating. Brier is the sum over five classes, averaged across cases. ECE is top-class calibration with 10 equal-width bins and is not a complete description of uncertainty. Intervals use inclusive rating categories and are conservative due to discretization.

`train_seconds` excludes JIT compilation, data parsing and KPI evaluation; MF includes validation/checkpoint/scale work. `site_memory_bytes` counts only cached EP message arrays, not peak process memory. `capped_updates` counts local updates that reach 10 iterations, and `skipped_updates` counts numerical failures; the reference has zero skipped updates.

Three seeded random splits overlap. Reported SD includes split and initialization variation; it is not a confidence interval, and a row-wise significance test would ignore user dependence. The report consequently avoids a statistical superiority claim.

The official raw data are deliberately absent. Only indices, derived model parameters and aggregate evidence are included. The package-level `MANIFEST.sha256` covers the final files, including documentation and plotting tools added after the numerical run.
