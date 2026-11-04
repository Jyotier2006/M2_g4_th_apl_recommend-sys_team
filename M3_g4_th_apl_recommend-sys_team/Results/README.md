# Measured experiment evidence

- `main`: full MovieLens 1M, seeds 42–46; 85 comparison rows.
- `robustness` plus `robustness_remaining`: 10 controlled conditions for each of
  seeds 42–44. The second folder completes an interrupted run without duplicates.
- `sensitivity`: prior, damping and order variants, seeds 42–44; 18 rows.
- `validation`: nine kernel/invariant checks, exact Gaussian/Gibbs/iterative
  diagnostic and bilinear quadrature. These small diagnostic scores are not
  MovieLens recommendation results.
- `analysis`: combined metrics, mean/sample SD, conditional paired user-bootstrap
  intervals, error groups, plots and `result_audit.json`.
- `m2_reference`: historical M2 evidence, labelled separately and not pooled
  into the fresh M3 timing summaries.

The result audit verifies all 229 rows, complete condition coverage, no duplicate
comparisons, finite KPIs, recorded validation selections, nonincreasing MAP
objectives, ADF/EP1 equality and main execution source hashes.

The main source snapshot and interrupted robustness core snapshot preserve the
executed model code. Each stage records its environment, configuration and hashes.
Final code adds recovery-only condition filtering and per-condition audit metadata;
numerical kernels and main selection logic are unchanged. Timings from resumed
robustness execution are not pooled into the main timing comparison.

Main seed-42 checkpoints and graph are retained. Per-user error sums support
reproducing conditional user-bootstrap intervals without distributing raw ratings
or all prediction vectors. Split permutations can be regenerated from the recorded
seeds and verified against their hashes. Raw MovieLens files are not bundled.

See Code/README.md for exact reproduction and analysis commands. Changing runs
requires rechecking the written scientific interpretation as well as figures.
