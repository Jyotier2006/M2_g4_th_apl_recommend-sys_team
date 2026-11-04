# Code locations for the TA

These line numbers refer to the exact source files in this package.

| Concept | File | Function / key line |
|---|---|---|
| Paired user resampling | `Code/src/analyze.py` | `bootstrap`, line 22 |
| Ordinal EP and bilinear VMP | `Code/src/bayes.py` | `local_messages`, line 48 |
| Cavity subtraction and site replacement | `Code/src/bayes.py` | `sweep`, line 114 |
| Latent variables and priors | `Code/src/bayes.py` | `init`, line 173 |
| Passes and damping | `Code/src/bayes.py` | `step`, line 187 |
| Five star probabilities | `Code/src/bayes.py` | `prob`, line 200 |
| EP pass / noise / prior sensitivity | `Code/src/experiments.py` | `run_ep`, line 59 |
| M2 SGD baseline | `Code/src/experiments.py` | `run_sgd`, line 84 |
| Shared splits, validation and stress tests | `Code/src/experiments.py` | `main`, line 150 |
| Genre vectors | `Code/src/gmrf.py` | `movie_info`, line 17 |
| Training-only movie graph | `Code/src/gmrf.py` | `graph`, line 33 |
| Neighbour-coupled MAP updates | `Code/src/gmrf.py` | `als`, line 91 |
| Validation-fitted residual scale | `Code/src/gmrf.py` | `calibrate`, line 138 |
| Pairwise Markov energy | `Code/src/gmrf.py` | `objective`, line 147 |
| Iterative Gaussian mean solver | `Code/src/inference_validation.py` | `gauss_seidel`, line 22 |
| Small Gaussian Gibbs sampler | `Code/src/inference_validation.py` | `gibbs`, line 31 |

Start with `gmrf.py`: show `graph`, then the `gamma * degree[j]` term and the neighbour sum in `als`, and then `objective`. The report gives the prior whose negative log produces these terms.

For EP, show `cp, ch = p - old_p, h - old_h` in `sweep`. This subtraction distinguishes repeated EP from counting the same rating again.

The exact source hash for each experiment is recorded in its result directory. Main and interrupted robustness snapshots are preserved; recovery-only filtering does not alter the numerical kernels.
