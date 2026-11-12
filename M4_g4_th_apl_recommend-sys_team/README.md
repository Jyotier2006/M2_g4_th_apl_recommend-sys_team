# M4 — Bayesian Personalized Recommendation using Latent Traits and EP

**CSE 516 · Theoretical AI · Group 4 · Final project**

| AU ID | Member | Role |
|---|---|---|
| AU2440170 | Jyotier Vithalani | Mathematical model, ADF/EP inference, repository |
| AU2440234 | Tanmay Kanani | Baselines |
| AU2440259 | Ansh Chaudhari | Literature and SOTA positioning |
| AU2440178 | Shlok Shah | Dataset and metadata |
| AU2440221 | Het Patel | KPIs, experiments and documentation |

We estimate a probability for each of the five possible MovieLens ratings.
The project tests how repeated approximate inference changes prediction
quality, and isolates the effect of an undirected movie graph by turning off
its smoothing term in the same MAP model.

On five MovieLens 1M splits, ADF, EP10, graph-free MAP and graph MAP attain
expected-rating RMSE of **0.8977, 0.8548, 0.8506 and 0.8499**, respectively.
The graph-only gain is **0.00064 stars** on clean data; gains are larger when
training ratings are removed or corrupted. The report covers split variation,
conditional confidence intervals, probability quality, runtime and limitations.

## Contents

| Folder | Contents |
|---|---|
| [Report](Report/README.md) | Final PDF report and its LaTeX source |
| [Video](Video/README.md) | Technical video walkthrough, transcript and captions |
| [Presentation](Presentation/README.md) | Final slides (PDF and PowerPoint) |
| [Code](Code/README.md) | Source, environment, configs, tests, experiment scripts and notebook |
| [Data](Data/README.md) | MovieLens 1M link, schema, preparation and license |
| [Results](Results/README.md) | Benchmark records, KPI tables, plots, bootstrap intervals and checks |

## Reproducing the results

From `Code/` with Python 3.12:

```bash
python -m pip install -r requirements-lock.txt
python tests/check.py
python tests/check_results.py --results ../Results
python src/final_kpis.py --results ../Results --out ../Results/final
```

The first two lines install the environment and run the nine model checks. The
last two rebuild the final KPI tables and plots from the recorded runs without
downloading the dataset. [Code/README.md](Code/README.md) has the commands for
rerunning all five splits (seeds 42–46), the robustness and sensitivity runs,
and the seed-42 replay.

## Earlier milestones

The main benchmark, robustness runs and model code continue from
[M3](https://github.com/Jyotier2006/M2_g4_th_apl_recommend-sys_team/tree/main/M3_g4_th_apl_recommend-sys_team).
M4 adds a fresh seed-42 replay, numerical checks, the final KPI summary, the
report, slides and walkthrough. The replay is a check and does not enter the
five-split means. M2 is at the root of this repository and M1 is in
[M1_g4_th_apl_recommend-sys_team](https://github.com/Jyotier2006/M1_g4_th_apl_recommend-sys_team).
