# Technical report and editable source

`M2_g4_th_apl_recommend-sys_team_Report.pdf` is the compiled report.

`latex/` is a self-contained source project:

- `main.tex`: report text and equations.
- `references.bib`: verified bibliographic entries.
- `generated/`: tables generated from experiment CSVs.
- `figures/`: vector PDF figures used in the report.

To use Overleaf, upload the **contents of `latex/`**, choose `main.tex` as the main document and use pdfLaTeX. To compile locally, run `python scripts/build_report.py` from `Code/`; this runs pdfLaTeX, BibTeX, and two further pdfLaTeX passes, then copies the compiled PDF to this directory.

The report records the actual experiment date. It explicitly distinguishes this fixed-threshold Matchbox variant from the full original system and distinguishes inherited M1 responsibility assignments from verified contributions.

Regenerate numerical assets with `python -m matchbox_m2.report_assets` after installing the code package. For a new configuration, also review the report's explanatory text and scope statements; automatically changing a table does not validate a new scientific conclusion.
