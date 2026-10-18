# CSE 516 - M2 Technical Foundation

## Bayesian Personalized Recommendation using Latent Trait Models & Expectation Propagation

**Group:** 4 | **Track:** Theoretical AI | **Snapshot:** 17 October 2026

| AU ID | Member |
|---|---|
| AU2440170 | Jyotier Vithalani |
| AU2440234 | Tanmay Kanani |
| AU2440259 | Ansh Chaudhari |
| AU2440178 | Shlok Shah |
| AU2440221 | Het Patel |

This folder continues the [M1 project](https://github.com/Jyotier2006/M1_g4_th_apl_recommend-sys_team). It contains a working, **restricted Matchbox implementation**, not a claim to reproduce all of Microsoft's original system. The research question is how single-pass ADF and repeated EP change predictive quality, uncertainty, computation and memory.

### What is complete

- Static, identity-only latent-trait model with Gaussian biases and fixed ordinal thresholds.
- Bayesian parameter updates using hybrid EP/VMP; proper cavity removal and replacement for repeated EP.
- MovieLens 1M downloader, SHA-256 validation, preparation and saved split indices.
- Global/user/item mean baselines, biased matrix factorization and separate ADF baseline.
- Full-data experiments for three seeds, two trait counts, and EP checkpoints 1/3/5.
- MAE, RMSE, NLL, Brier, ECE, predictive intervals, fitting/prediction timings and iteration diagnostics.
- Nine passing mathematical/implementation checks plus independent bilinear quadrature evidence.
- Compiled PDF report, editable LaTeX/BibTeX, figures, raw metric CSVs, logs, environment and a fitted model checkpoint.

### Main measured result

All values below are mean ± sample SD across seeds 42, 43 and 44. Each run uses 800,167 train / 100,020 validation / 100,022 test ratings.

| Method | Test RMSE | Test NLL | Fitting time |
|---|---:|---:|---:|
| ADF, K=8 | 0.8980 ± 0.0049 | 1.2520 ± 0.0046 | 2.644 ± 0.212 s |
| EP5, K=8 | 0.8633 ± 0.0040 | 1.2176 ± 0.0032 | 13.632 ± 0.879 s |
| MF, categorical mean | 0.8681 ± 0.0031 | 1.2210 ± 0.0033 | 3.000 ± 0.565 s |
| MF, conventional clipped point | 0.8646 ± 0.0031 | Same fitted probability wrapper | Same fitted model |

EP improves over ADF, but the comparison with conventional MF is close; MF wins on one seed and fits faster. EP's ECE gets worse, even as NLL improves. Three overlapping splits do not establish statistical superiority. See the report for the approximation errors, timing conventions and limitations.

### Contents

| Path | Purpose |
|---|---|
| `Report/M2_g4_th_apl_recommend-sys_team_Report.pdf` | Compiled technical report |
| `Report/latex/` | Self-contained LaTeX source, BibTeX, generated tables and PDF figures |
| `Code/src/matchbox_m2/` | Model, learning, inference, baselines, data, metrics, graph, validation and CLI |
| `Code/configs/m2.json` | Prespecified reference configuration |
| `Code/README.md` | Exact setup and run instructions |
| `Data/README.md` | Official dataset source, download and split protocol |
| `Results/reference/` | Actual 17 October runs, CSVs, plots, logs, indices, model and provenance |
| `Video/README.md` | Instructions for adding the team's required video |
| `SUBMISSION_CHECKLIST.md` | Mapping from M2 requirements to evidence |
| `SOURCES_AND_SCOPE.md` | Source audit and changes since M1 |
| `TEAM_REVIEW.md` | Member review tasks; no invented contribution claims |
| `DEFENSE_NOTES.md` | Short technical explanations for review |

### Run it

From `Code/`, after creating a Python 3.12 virtual environment:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m matchbox_m2.validation
python -m matchbox_m2.demo
python scripts/reproduce.py
```

The complete reproduction downloads MovieLens from GroupLens if necessary and writes to `Results/reproduced/`. It refuses to overwrite an earlier experiment. The supplied reference results remain intact. See `Code/README.md` for Windows activation, individual commands, checkpoint inference and PDF building.

### Before uploading

Upload the **extracted folder**, preserving its name `M2_g4_th_apl_recommend-sys_team`, to the course-designated repository. Do not replace M1. The supplied M1 repository is a reference; verify the course's designated M2 upload destination.

The team must add its video file or real video link, create its own PPT as requested, review and reproduce the technical work, and each member must submit the separate Google Form. Those items are not claimed as completed. The M2 notice says 15 October 2026 at 11:59 PM; the general guideline PDF lists 11 October. The actual work here is dated 17 October; any extension must be confirmed with the course team.

### Scope and attribution

Metadata, learned per-user thresholds, dynamics, controlled cold-start and full robustness studies remain M3 work. PreRec is a conceptual SOTA anchor, not an executed baseline. This package was developed with generative-AI assistance, including code, experiment execution and report preparation; the students must review and be able to explain it.

References: [Matchbox, WWW 2009](https://doi.org/10.1145/1526709.1526725), [PreRec, WSDM 2024](https://doi.org/10.1145/3616855.3635779), [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/), [MovieLens history and context](https://doi.org/10.1145/2827872).
