"""Consolidate recorded runs and optionally verify a fresh run of split 42."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = {
    "Global": "Global", "UserMean": "User mean", "ItemMean": "Item mean",
    "MF_SGD": "MF-SGD", "ADF": "ADF / EP1", "EP3": "EP3", "EP5": "EP5",
    "EP10": "EP10", "MF_MAP": "MF-MAP", "GMRF_MAP": "GMRF-MAP",
}
FIELDS = ["rmse", "mae", "nll", "brier", "ece", "star_accuracy",
          "fit_seconds", "predict_ms_per_1k"]


def tex_table(path, headers, rows):
    bs = chr(92)
    lines = [bs + "begin{tabular}{@{}l" + "r" * (len(headers) - 1) + "@{}}" + bs + "toprule",
             " & ".join(headers) + bs * 2 + bs + "midrule"]
    lines.extend(" & ".join(str(x) for x in row) + bs * 2 for row in rows)
    lines.append(bs + "bottomrule" + bs + "end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(recorded, recorded_dir, rerun, out):
    def split(directory):
        audits = json.loads((directory / "split_audits.json").read_text())
        return next(a for a in audits if a["seed"] == 42)
    a, b = split(recorded_dir), split(rerun)
    if a["split_hashes"] != b["split_hashes"] or a["counts"] != b["counts"]:
        raise ValueError("Replay train, validation or test split differs from the benchmark.")
    ref = recorded[recorded.seed.eq(42)].set_index("method").sort_index()
    cur = pd.read_csv(rerun / "metrics.csv").set_index("method").sort_index()
    if not ref.index.equals(cur.index):
        raise ValueError("Rerun does not have the same methods as split 42.")
    rows = []
    for field in ("train_n", "test_n", "rmse", "mae", "nll", "brier", "ece",
                  "star_accuracy", "validation_rmse", "gamma", "epoch"):
        if not ref[field].isna().equals(cur[field].isna()):
            raise ValueError(f"Missing values differ for {field}")
        for method in ref[field].dropna().index:
            a, b = float(ref.at[method, field]), float(cur.at[method, field])
            rows.append(dict(method=method, metric=field, recorded=a, rerun=b,
                             absolute_difference=abs(a-b)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "seed42_replication.csv", index=False)
    delta = float(df.absolute_difference.max())
    audit = {"seed": 42, "methods": len(ref), "compared_values": len(df),
             "max_absolute_difference": delta, "tolerance": 1e-8,
             "split_hashes_matched": True, "timings_compared": False, "passed": delta <= 1e-8}
    (out / "seed42_replication.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not audit["passed"]:
        raise ValueError(f"Replication difference exceeds tolerance: {delta}")
    print(json.dumps(audit))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rerun", type=Path)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.results / "main" / "metrics.csv")
    tab = table.loc[table.method.isin(METHODS)].copy()
    if tab.duplicated(["method", "seed"]).any() or not tab.groupby("method").seed.agg(set).map(
            lambda s: s == set(range(42, 47))).all() or set(tab.method) != set(METHODS):
        raise ValueError("Expected exactly one observation per displayed method and seed 42--46.")
    tab["predict_ms_per_1k"] = tab["predict_seconds"] * 1e6 / tab["test_n"]
    stats = tab.groupby("method")[FIELDS].agg(["mean", "std"])
    flat = stats.copy()
    flat.columns = [f"{metric}_{op}" for metric, op in flat.columns]
    flat = flat.reindex(METHODS)
    flat.to_csv(out / "main_kpis.csv")
    def cell(name, key, digits=4, scale=1):
        m, s = stats.loc[name, key]
        return f"{m*scale:.{digits}f} $" + chr(92) + f"pm$ {s*scale:.{digits}f}"
    tex_table(out / "quality.tex", ["Method", "RMSE", "NLL", "Brier", "ECE", "Stars (\\%)"],
              [[label, *[cell(name, key) for key in ["rmse", "nll", "brier", "ece"]],
                cell(name, "star_accuracy", 2, 100)] for name, label in METHODS.items()])
    names = ["MF_SGD", "ADF", "EP5", "EP10", "MF_MAP", "GMRF_MAP"]
    tex_table(out / "cost.tex", ["Method", "Fit (s)", "Predict (ms / 1000)", "MAE"],
              [[METHODS[n], cell(n,"fit_seconds",2), cell(n,"predict_ms_per_1k",3),
                cell(n,"mae")] for n in names])
    g = pd.read_csv(args.results / "main" / "graph_stats.csv")
    g[["seed","movies","edges","density","mean_degree","isolated_movies",
       "scalar_trait_bias_variables","graph_seconds"]].to_csv(out / "representation.csv", index=False)
    nrows = ["movies","edges","density","mean_degree","scalar_trait_bias_variables"]
    g[nrows].agg(["mean","std"]).T.to_csv(out / "representation_summary.csv")
    # Timings are from the preserved five runs; no timing from the rerun enters this figure.
    fig, ax = plt.subplots(figsize=(7.4, 3.7), layout="constrained")
    shifts = {"MF_SGD":(6,6),"ADF":(-38,7),"EP5":(7,7),"EP10":(7,-14),
              "MF_MAP":(-47,8),"GMRF_MAP":(6,-15)}
    for i, n in enumerate(names):
        x, dx = stats.loc[n,"predict_ms_per_1k"]
        y, dy = stats.loc[n,"rmse"]
        ax.errorbar(x,y,xerr=dx,yerr=dy,fmt="o",capsize=3,color=f"C{i}")
        ax.annotate(METHODS[n],(x,y),xytext=shifts[n],textcoords="offset points",fontsize=9)
    ax.set(xlabel="Batch prediction time (milliseconds per 1,000 ratings)",
           ylabel="Test RMSE (stars)", title="Inference cost versus predictive error")
    ax.grid(alpha=.25)
    ax.margins(x=.24,y=.20)
    fig.savefig(out / "inference_tradeoff.pdf")
    fig.savefig(out / "inference_tradeoff.png", dpi=190)
    plt.close(fig)
    if args.rerun:
        compare(table, args.results / "main", args.rerun, out)


if __name__ == "__main__":
    main()
