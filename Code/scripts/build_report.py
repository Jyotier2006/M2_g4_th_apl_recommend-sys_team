"""Compile the supplied LaTeX report, resolving BibTeX and cross-references."""
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
latex = root / "Report/latex"
for program in ["pdflatex", "bibtex"]:
    if not shutil.which(program):
        raise SystemExit(f"Install {program} via TeX Live/MiKTeX, or use Overleaf with Report/latex")
for command in [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]]:
    result = subprocess.run(command, cwd=latex, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)
log = (latex / "main.log").read_text(errors="replace")
if "Overfull \\hbox" in log or "undefined references" in log or "undefined citations" in log:
    raise SystemExit("Review main.log: overflow or unresolved references remain")
dest = root / "Report/M2_g4_th_apl_recommend-sys_team_Report.pdf"
shutil.copy2(latex / "main.pdf", dest)
print(f"Compiled report: {dest}")
