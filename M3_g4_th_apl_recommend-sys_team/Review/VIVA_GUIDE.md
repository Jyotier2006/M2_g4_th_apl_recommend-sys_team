# How to explain M3

**Problem:** Predict a user's missing 1–5 movie rating from observed ratings, and
measure whether the predicted probabilities are useful and calibrated.

**What M2 did:** A restricted static Matchbox latent-trait factor graph, with
ADF and repeated EP/VMP updates, plus mean and matrix-factorization baselines.

**What M3 adds:** More passes and seeds; a movie-to-movie Gaussian Markov prior;
a matched graph-off ablation; missing-data, label-noise, graph-density and genre
tests; probability-quality analysis; paired user-bootstrap intervals; a small
exact/Gauss–Seidel/Gibbs diagnostic; and an updated report based on measured runs.

## Important terms

| Term | Meaning in this project |
|---|---|
| Latent trait | An unnamed learned numeric coordinate of a user/movie vector; not a fixed genre |
| Bias | A user/movie tendency to give/receive higher ratings |
| ADF | One sequential pass; the rating contribution is not retained as a revisitable site |
| EP site | A stored Gaussian approximation to one rating's contribution |
| Cavity | Current natural parameters minus the old site's parameters |
| EP1–EP10 | Total passes through the same training ratings, not ten datasets |
| VMP | Variational message passing used for the bilinear product messages |
| Markov network | Undirected probabilistic dependencies represented by factors |
| GMRF | A Gaussian Markov random field; precision zeros encode conditional independence |
| MAP | Most probable parameter configuration under the likelihood and prior |
| Gamma | Strength of the penalty for neighbouring movie vectors disagreeing |
| Laplacian | `L = D - W`; turns pairwise similarity penalties into a quadratic form |
| Observation noise | Unexplained variation between latent affinity and observed score |
| Corrupted labels | Artificially altered training ratings used for a stress test |
| Calibration | Whether stated prediction confidence agrees with observed correctness |
| Ablation | Removing one component, such as setting gamma=0, to measure its contribution |
| Seed | Initial state of the pseudorandom generator, used to reproduce a split/initialization |

## Where the Markov network is implemented

In `Code/src/gmrf.py`:

1. `movie_info` builds multi-hot genre vectors.
2. `graph` constructs training-only residual similarities, mixes genres, selects
   neighbours, symmetrizes, and normalizes the edge matrix.
3. `als` adds `gamma * degree[j]` to a movie's trait precision block and
   `gamma * weight * neighbour_trait` to its right-hand side.
4. `objective` calculates the graph quadratic penalty, equivalent to a sum of
   weighted squared differences across undirected edges.

The graph factor is `exp(-gamma*w_ij*||q_i-q_j||^2/2)`.
With positive ridge precision, `J = lambda*I + gamma*(D-W)` is positive definite.
This is the probabilistic justification for the regularizer.

## Questions likely to come up

**Why can the graph help?** It shares evidence between similar movies and can
reduce variance. Wrong neighbours or excessive strength can cause oversmoothing.
Use the actual ablation results to say where the hypothesis was supported.

**Is GMRF versus EP a clean test of the graph?** No. The likelihood and fitting
method also change. The clean comparison is GMRF-MAP versus the same MAP model
with gamma=0.

**Is EP always better than ADF?** No. On these full-data runs, repeated passes
can improve error, at extra time and site memory. The earlier synthetic example
did not prove the same ordering.

**Is a Bayesian model the same as a Bayesian network?** No. Bayesian learning
uses priors and posteriors; a Bayesian network specifically uses a directed
acyclic graph. Our explicit new network is an undirected GMRF.

**What does the Gibbs experiment establish?** It checks approximate inference
against an exactly solvable small Gaussian distribution. It is not a full-data
MovieLens Gibbs recommender and does not establish an exact bilinear posterior.

**Why not report SHD?** We do not know a true movie-dependency graph. A similarity
graph cannot be assigned meaningful structural recovery accuracy without one.

**What remains limited?** Random interaction splits, overlapping seeds, fixed
global thresholds, approximate EP posteriors, MAP residual calibration without
parameter uncertainty, nonrandom real-world missingness, and no causal or
cross-domain evaluation.

**Is this PreRec?** No. PreRec is the research-lab SOTA reference. Its neural
encoders, domain/popularity variables, and causal debiasing are not implemented.

**What should the conclusion say?** Read the measured results, compare like
with like, state the size and uncertainty of the graph effect, and name cases
where it does not help. Do not claim a universal improvement.
