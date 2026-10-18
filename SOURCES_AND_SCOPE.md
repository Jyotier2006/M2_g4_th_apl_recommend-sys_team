# Source audit and scope decisions

## Material reviewed

| Source | What it establishes |
|---|---|
| `G4_PP2026_CSE516 (1).pdf` | M0 proposal, Group 4 members, Theoretical AI, Matchbox/PreRec references and MovieLens 1M plan |
| `PDF_CSE 516 Project Guidelines (2).pdf` | Mandatory representation-learning-inference framework, academic depth, validation, baselines, SOTA, reproducibility and AI accountability |
| `L5_Project Guidelines (2).pdf` | Lecture slides and quiz emphasizing the same research and evaluation expectations |
| `image.png` through `image(5).png` | M2-specific objectives, required report/video/code/results, folder names, minimum files and date |
| `image(6).png` | Group registry: G4, five members, Theoretical category and Matchbox title/DOI |
| M1 report | Research question, scope, five planned experiments, baseline/KPI plan and responsibility assignments |
| M1 root/Data/Results READMEs | No implementation or measured results claimed at M1 |
| M1 draft diagram | Planned traits, biases, metadata and thresholds |
| Stern et al., WWW 2009 | Primary Matchbox formulation and hybrid EP/VMP plus ADF/EP scheduling |
| Lin et al., WSDM 2024 / arXiv v4 | PreRec model, MAP/Adam learning, causal/transfer framing and its distinct ranking evaluation |
| Official GroupLens archive/README | Actual counts, file schema, checksum-verified benchmark and redistribution terms |

## M1 repository snapshot

URL: https://github.com/Jyotier2006/M1_g4_th_apl_recommend-sys_team

Commit: `16a72dc91e1acb3342c411ac264ceb9eeebe9856`

Files present: root README; `Data/README.md`; `Report/M1_g4_th_apl_recommend-sys_team_Report.pdf`; an empty Report README; `Results/Matchbox_PGM_Draft.png`; Results README; and a text file linking a Drive video. No PPT, code, notebook, data, or experimental metrics were present in this snapshot. The video link was read, but the video's contents were not reviewed or used as evidence. The report and all other available substantive files were read.

M0 refers to an older `CSE-516_GROUP-4` repository. This task supplied the separate M1 repository above. This ZIP does not assume an unverified M2 remote or mutate either repository.

## Original papers

1. D. H. Stern, R. Herbrich, T. Graepel. **Matchbox: Large Scale Online Bayesian Recommendations.** WWW 2009, pp. 111-120. DOI: https://doi.org/10.1145/1526709.1526725. Author-hosted full text: https://www.herbrich.me/papers/www09.pdf.
2. Z. Lin, H. Ding, N. T. Hoang, B. Kveton, A. Deoras, H. Wang. **Pre-trained Recommender Systems: A Causal Debiasing Perspective.** WSDM 2024, pp. 424-433. DOI: https://doi.org/10.1145/3616855.3635779. Author preprint: https://arxiv.org/abs/2310.19251 (v4, 8 January 2024). The longer preprint contains supplementary implementation details.

The papers and course materials were used as references but are not repackaged as our own deliverables. Bibliographic records are in `Report/latex/references.bib`.

## What changed since M1

The central question and dataset remain unchanged. M2 isolates an identity-only, static, fixed-threshold instance rather than claiming every Matchbox extension is already implemented. Repeated EP uses cached sites and cavity removal; the product step uses VMP. This is a disclosed implementation variant rather than the original Infer.NET executable.

The M2 main experiment is random interaction prediction, not the paper's user cold-start protocol. Published paper scores consequently are not mixed with new measured scores. Metadata is validated but unused in training; learned thresholds, metadata ablations and controlled sparsity/cold-start remain M3. PreRec is compared conceptually and is not represented as an executed competitor or an EP model.

## Result provenance and responsibility

All benchmark numbers were produced by code during this preparation on 17 October 2026. The initial M1 roles were preserved as assignments, not fabricated as completed individual work. AI assisted code, execution and documentation. The team must review, reproduce and explain the package before submission, consistent with the provided course guidance.
