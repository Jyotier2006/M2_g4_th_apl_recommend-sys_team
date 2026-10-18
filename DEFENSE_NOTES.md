# Technical review notes

## What is the actual research question?

How much predictive and probabilistic quality do additional EP passes buy, and what computation/memory do they cost? Building a recommender is the benchmark; the theoretical subject is approximate inference.

## Why use a factor graph?

It separates priors, trait products, addition, observation noise and ordinal feedback into identifiable local factors. A rating couples one user's traits and biases with one item's traits and biases. Shared parameters transmit information across ratings.

## What is learned?

Approximate Gaussian posterior means/variances for user traits, item traits and biases. The graph structure, global thresholds, observation variance and prior variances are fixed. The global offset is the training mean. We do not learn causal structure.

## ADF and EP in plain terms

ADF incorporates each rating once and discards its individual message. Later ratings update the model, but earlier ratings are not revisited. EP saves those individual messages. Before revisiting a rating, it subtracts that rating's previous precision and precision-weighted mean, recomputes its contribution using the remaining evidence, and installs the replacement.

Gaussian multiplication is addition of natural parameters. Gaussian division is subtraction. That is why the cached-site identity can be tested directly.

## Why not just loop over the data five times with ADF?

The posterior already contains the earlier contribution. Updating it again without removing the old contribution treats repeated evidence as new evidence. The implementation disallows a second ADF pass. EP1 and ADF give exactly the same posterior in all six full-data comparisons because initial sites are zero and all other conditions match.

## Is every update exact EP?

No. Matchbox combines approximate methods. Our product factor uses VMP reverse messages involving `E[t²]`, while forward moments and ordinal EP projection handle the other steps. Gaussian product projection, the factorized posterior and the fixed-threshold simplification introduce approximation error. This is not the complete original Infer.NET implementation.

## What does a prediction return?

Five probabilities, one for each rating. Trait/bias means give a latent score mean; posterior variances and observation noise give its approximate variance. Gaussian mass between fixed cut points yields rating probabilities. The expected rating minimizes squared error under that distribution; the median minimizes absolute error. Both decisions are recorded.

## What did the checks establish?

The formulas agree with independent integration in tractable cases, the global/site identity is maintained, a single rating is not counted repeatedly by EP, probabilities are normalized, split rows are disjoint and complete, and saved models reload consistently. A tiny bilinear quadrature experiment also shows what the approximation loses: variances are around 10% too small and a real posterior correlation is dropped. No test proves global EP convergence.

## What is the honest result?

At K=8, RMSE improves from about 0.8980 with ADF to 0.8633 with EP5. Fitting time grows from about 2.64 s to 13.63 s, and EP stores about 219.77 MiB of rating messages. Conventional MF has RMSE about 0.8646 with its standard clipped prediction and fits in about 3.00 s. It wins on one seed. We cannot claim a decisive improvement over MF.

## Did uncertainty improve?

Some measures improved: EP5 has better NLL and Brier than ADF. Top-class ECE worsened. The total rating distribution also includes fixed observation noise and ordinal discretization, so local posterior variance underestimation does not automatically imply overconfident rating probabilities. The reported rating intervals are broad and conservative.

## Why is PreRec in the report?

It is the M1-selected modern research anchor for Bayesian recommendation and limited-data transfer. Its domain/popularity variables and neural encoders address questions that our static within-domain rating model does not. It uses MAP/Adam learning and a different ranking evaluation. We did not run it, and its scores cannot be compared directly with our MovieLens RMSE.

## What is not finished?

Learned user thresholds, metadata experiments, controlled user cold-start/sparsity, temporal evaluation, a larger sensitivity study and more rigorous statistical analysis. Those are M3 directions. The team also supplies the video/PPT and individual forms.
