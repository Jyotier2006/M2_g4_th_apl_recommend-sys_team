# Final report

- [M4_g4_th_apl_recommend-sys_team_Report.pdf](M4_g4_th_apl_recommend-sys_team_Report.pdf) is the compiled report.
- [LaTeX/main.tex](LaTeX/main.tex) is the editable report. Its local
  bibliography, figures and generated tables are bundled alongside it.

From this directory, build using TeX Live or MiKTeX with `latexmk`:

```bash
cd LaTeX
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For Overleaf, upload the contents of `LaTeX/` and select `main.tex` and pdfLaTeX.
The report tables come from `Code/src/final_kpis.py` and `Code/src/analyze.py`.
The factor graph is the original model diagram. The movie subgraph, validation
curves and bootstrap intervals, robustness and reliability plots come from the
recorded analysis; the prediction-time plot comes from the final KPI script.
See [the run guide](../Code/README.md) to regenerate them.
