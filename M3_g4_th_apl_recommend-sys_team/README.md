# M3 — Bayesian Personalized Recommendation

**Group 4 · CSE 516 · Theoretical AI · Results and Analysis**

This is the cumulative M3 update of the M2 report. It retains the original
latent-trait model and EP/VMP foundation, and adds an explicit Gaussian Markov
movie network, multiple-method experiments, robustness, statistics, and analysis.

Open `Report/M3_g4_th_apl_recommend-sys_team_Report.pdf` first.

## Main measured result

Full official MovieLens 1M; five random 80/10/10 splits; mean ± sample SD:

| Method | Expected-rating RMSE |
|---|---:|
| ADF / EP1 | 0.897727 ± 0.003510 |
| EP5 | 0.864335 ± 0.003717 |
| EP10 | 0.854804 ± 0.002898 |
| MF-MAP, graph disabled | 0.850576 ± 0.002060 |
| GMRF-MAP | 0.849932 ± 0.002298 |

The isolated graph gain is **0.000644 RMSE**, about **0.076%**. It is small:
four split-specific user-bootstrap intervals lie above zero; one includes zero.
Comparing GMRF with EP also changes the likelihood and fitting method, so the
whole improvement over EP must not be attributed only to the network.
Noise and missing-data tests provide further evidence about when smoothing helps.
These are real-data runs, not the earlier synthetic offline demo.

## Contents

| Folder | Contents |
|---|---|
| Report | Updated PDF, editable LaTeX, references and generated tables |
| Code | Procedural source, configurations, meaningful validation checks and run instructions |
| Data | Official dataset source, preparation and field mapping; no raw ratings bundled |
| Results | Raw metrics, traces, audits, logs, checkpoints, summaries, bootstrap intervals and plots |
| Review | M2-to-M3 changes, guideline coverage, viva explanations, code locations and inherited team roles |

Quick start from `Code`:

```bash
python -m pip install -r requirements.txt
python tests/check.py
python tests/check_results.py --results ../Results
```

See `Code/README.md` for full experiments, figure regeneration and checkpoint
queries. Raw MovieLens data download automatically when needed.

## Submission

Add this complete folder to the **same existing repository**:
https://github.com/Jyotier2006/M2_g4_th_apl_recommend-sys_team

Do not create a new repository. The M3 notice requires the report and root README;
the included code/results support the evidence. No M3 video is required.
Every member must submit the individual Google Form separately and confirm
their actual contribution. This package does not claim those submissions or
member reviews have already happened.

## Scope and honesty

The graph is a proper Gaussian Markov prior, not a causal graph or neural network.
EP is approximate; MAP probabilities use validation-fitted residual noise rather
than a full parameter posterior. Exact/Gibbs results are explicitly labelled as
small Gaussian diagnostics. Matchbox is the implementation anchor and PreRec is
a conceptual research-lab reference; neither full system is reproduced here.
