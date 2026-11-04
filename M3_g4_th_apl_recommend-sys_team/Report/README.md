# Updated cumulative M3 report

`M3_g4_th_apl_recommend-sys_team_Report.pdf` is the submission report.
The editable source is `latex/main.tex` with the accompanying section files,
BibTeX references, original factor graph and generated tables.

This updates the revised M2 report, retaining its project/team identity,
probabilistic formulation, EP/VMP derivation and validation foundation.
The new experimental setup, results and conclusions refer to actual M3 runs.

To rebuild after generating analysis figures, enter `latex` and run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The report imports plots from `../../Results/analysis/plots`; keep the folder
structure intact.
