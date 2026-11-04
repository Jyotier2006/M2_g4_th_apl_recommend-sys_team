# Changes from M2 to M3

The report continues the existing M2 LaTeX report. Its project identity, team,
latent-trait equations, EP/VMP derivation, and numerical-validation foundation
are retained. Experimental setup, results, limitations and conclusion are updated;
a Markov-network section, new figures and statistical analysis are added.

| M2 feedback / gap | M3 change | Evidence |
|---|---|---|
| Too few approaches | Mean predictors, original SGD MF, ADF, EP1–5/10, alternating MAP MF, GMRF MAP | Results/main/metrics.csv |
| Explicit Markov or Bayesian network | Proper Gaussian Markov movie prior with pairwise factors and neighbour-coupled block solves | Code/src/gmrf.py; report Markov section |
| Explain why output improves | Matched gamma=0 ablation; separate graph contribution from optimizer/likelihood changes; report effect sizes and conditional user-bootstrap intervals | Results/analysis/paired_user_bootstrap.csv |
| Noise not studied | EP observation variances 0.5/1/2 and controlled 10%/20% training-label corruption | Results/main and Results/robustness |
| Single-condition evidence | Five main seeds; three robustness seeds; data availability, graph density, feature and cold-start tests | Code/configs; Results |
| Theory and convergence | Exact Gaussian versus Gauss–Seidel/Gibbs diagnostic; bilinear quadrature; objective and EP change traces | Results/validation; Results/analysis/plots |
| Missing probability/statistical analysis | NLL, Brier, ECE, reliability, bootstrap, support/class errors | Results/analysis |
| Report continuity | Original mathematical foundation retained; cumulative report revised rather than a separate unrelated report | Report/latex/main.tex |

The three core EP kernels (truncation, local messages, sweep) are unchanged.
Optional prior/noise/damping arguments expose sensitivity tests; default settings
still reproduce original M2 natural parameters exactly. The original M2 SGD
numerical kernel is unchanged and wrapped with procedural dictionary state.

The main experiment's exact source snapshot is retained in
Results/main/source_snapshot. After that run, experiments.py gained a controlled
cold-user condition and per-condition audit metadata for the robustness run.
These changes do not modify the main model, kernels, split or selection logic.
Final source hashes and each run's source hashes are included for traceability.

Scope limits remain explicit: no PreRec implementation, no causal inference,
no learned user-specific thresholds, no full-data Gibbs posterior, and no
proof that a graph universally improves recommendation.
