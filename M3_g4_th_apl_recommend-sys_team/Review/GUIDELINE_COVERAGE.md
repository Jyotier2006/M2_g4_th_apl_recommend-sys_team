# M3 requirements and evidence

Reviewed sources: all six supplied M3 screenshots; `PDF_CSE 516 Project
Guidelines (2).pdf`; `L5_Project Guidelines (2).pdf`, including image-only slides.
The M3-specific notice overrides the older general video requirement and deadline.

| Requirement | Package evidence / action |
|---|---|
| Update previous report | Existing M2 LaTeX foundation retained; updated PDF and editable LaTeX in Report |
| Same GitHub repository | Add this M3 folder to Jyotier2006/M2_g4_th_apl_recommend-sys_team; no new repo |
| Correct naming | M3_g4_th_apl_recommend-sys_team; report in Report/ |
| Root README | README.md explains contents and reproduction entry points |
| Full experiments and common conditions | Five main seeds; fixed paired splits; validation-only selection; configs and audits |
| Compare with M2 baselines | Global/user/item means, original SGD, ADF, EP; added MAP and graph ablation |
| Different approaches | ADF, cached-site EP, SGD, alternating MAP; exact Gaussian/Gauss–Seidel/Gibbs small-model diagnostic |
| Explicit network | Proper GMRF with nodes, edges, factors, precision, conditionals and MAP derivation |
| Explain claimed improvement | Small graph-only effect separated from full-system EP comparisons; uncertainty and failures reported |
| KPI plots/tables | Prediction/probability comparisons, reliability, cost, convergence, robustness, errors |
| Statistical variation | Five main and three robustness seeds; mean ± sample SD |
| Statistical support | Split-specific paired user-cluster bootstrap CIs; limits of overlapping splits explained |
| Probability quality | NLL, Brier, ECE, exact-star accuracy, probability vectors, rating intervals |
| Noise | EP likelihood variance sensitivity plus controlled training-label corruption |
| Missingness/sample size | Nested retained-training fractions with common evaluation rows |
| Graph density | Top-neighbour settings 5/20/50; actual symmetric edge statistics |
| Ablation | Gamma=0, positive graph, rating-only, genre-only graphs |
| Error analysis | True rating and item/user-support groups; controlled cold items/users |
| Theoretical track | Equations, invariants, quadrature, exact-reference comparison, complexity and convergence |
| Explain results | Report discusses expected mechanisms, evidence, negative results and limitations |
| Revised SOTA | Specific Matchbox and PreRec relationship; no full reproduction or unsupported SOTA claim |
| Code README | Commands, environment, seeds, configuration, execution scope |
| Data README | Official source, preparation, fields and checksum verification; no raw data redistribution |
| Logs and provenance | Results include raw metrics, configs, environment, hashes, run logs and source snapshot |
| No video for M3 | No video deliverable required by the supplied M3 notice |
| Individual Google Form | Each member must submit separately; not completed by this package |
| Individual accountability | Historical M1/M2 roles retained; actual M3 contributions remain for member confirmation |
| Responsible AI use | Explicit assistance disclosure; results produced by execution; students must verify/understand |

SHD/SID and structure-recovery precision are not reported because no true movie
dependency graph is available. They are examples in the guidelines, not valid
metrics to invent for this rating-prediction task.

Before submission, every member should inspect the report and reproduce the
checks relevant to their role. The package does not certify that member review,
the individual form, or a GitHub submission has already happened.
