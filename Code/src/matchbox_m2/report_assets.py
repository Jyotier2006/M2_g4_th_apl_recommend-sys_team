"""Generate figures and LaTeX tables directly from the saved experiment CSVs."""
import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

from .data import write_json
from .graph import build_graph
from .validation_study import run_study

ROOT = Path(__file__).resolve().parents[3]
COLORS = {"blue":"#246482", "orange":"#bd6b2d", "green":"#42816c", "gray":"#66717b"}


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def graph_figure(path):
    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8.2); ax.axis("off")
    variables = {"s":(1.8,6.3), "t":(4.5,6.3), "bu":(7.5,6.3), "bi":(10.,6.3),
                 "z":(3.15,4.4), "a":(6.1,2.7), "y":(6.1,1.1), "R":(10.1,1.1)}
    factors = {"ps":(1.8,7.6), "pt":(4.5,7.6), "pb":(7.5,7.6), "pi":(10.,7.6),
               "prod":(3.15,5.35), "sum":(6.1,3.6), "noise":(6.1,1.9), "ord":(8.1,1.1)}
    connections=[("ps","s"),("pt","t"),("pb","bu"),("pi","bi"),
                 ("prod","s"),("prod","t"),("prod","z"),("sum","z"),
                 ("sum","bu"),("sum","bi"),("sum","a"),("noise","a"),("noise","y"),
                 ("ord","y"),("ord","R")]
    for f,v in connections:
        ax.plot([factors[f][0],variables[v][0]],[factors[f][1],variables[v][1]],color="#73828b",lw=1.5,zorder=1)
    labels={"s":r"$s_{uk}$","t":r"$t_{ik}$","bu":r"$b_u$","bi":r"$b_i$",
            "z":r"$z_{uik}$","a":r"$a_{ui}$","y":r"$y_{ui}$","R":r"$R_{ui}$"}
    for n,(x,y) in variables.items():
        ax.add_patch(Circle((x,y),.34,facecolor="#d9e7ef" if n=="R" else "white",edgecolor=COLORS["blue"],lw=1.5,zorder=3))
        ax.text(x,y,labels[n],ha="center",va="center",fontsize=12,zorder=4)
    flabels={"ps":"N","pt":"N","pb":"N","pi":"N","prod":r"$\times$","sum":r"$\Sigma$","noise":"N","ord":"I"}
    for n,(x,y) in factors.items():
        ax.add_patch(Rectangle((x-.22,y-.22),.44,.44,facecolor=COLORS["blue"],zorder=3))
        ax.text(x,y,flabels[n],color="white",ha="center",va="center",fontsize=12,zorder=4)
    ax.add_patch(Rectangle((.7,3.95),4.9,4.12,fill=False,linestyle="--",edgecolor="#8fa0ad",lw=1))
    ax.text(.9,4.12,r"$k=1,\ldots,K$",fontsize=10,color="#54646e")
    ax.text(6.55,3.6,r"$a=\mu+b_u+b_i+\sum_k z_k$",fontsize=10,va="center")
    ax.text(6.6,1.9,r"noise variance $\beta^2$",fontsize=9,va="center")
    ax.text(8.1,.45,"fixed thresholds",ha="center",fontsize=9)
    ax.text(1.8,.9,"Circles: random variables\nSquares: factors\nShading: observed rating",fontsize=10,ha="center",va="center")
    ax.text(6,8.4,"One rating factor; user and item parameters are shared across ratings",ha="center",fontsize=11)
    save(fig,path)


def run(results, latex):
    figures = results / "plots"; figures.mkdir(exist_ok=True)
    tables = latex / "generated"; tables.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold","pdf.fonttype":42})
    metrics=pd.read_csv(results/"metrics.csv")
    summary=pd.read_csv(results/"summary.csv")
    trace=pd.read_csv(results/"learning_trace.csv")
    run_study(results)
    write_json(results/"graph_instance.json",build_graph())
    graph_figure(figures/"factor_graph")
    chosen=[("global_mean",0,1,"Global mean"),("user_mean",0,1,"User mean"),
            ("item_mean",0,1,"Item mean"),("mf",8,25,"MF, K=8"),
            ("adf",8,1,"ADF, K=8"),("ep",8,3,"EP3, K=8"),("ep",8,5,"EP5, K=8")]
    selected=[]
    for method,k,passes,label in chosen:
        row=summary[(summary.method==method)&(summary.k==k)&(summary.passes==passes)].iloc[0]
        selected.append((row,label))
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for ax,metric,title in zip(axes,["rmse","nll"],["Rating error (lower is better)","Predictive log-loss (lower is better)"]):
        y=np.arange(len(selected))
        ax.errorbar([r[metric+"_mean"] for r,_ in selected],y,
                    xerr=[r[metric+"_std"] for r,_ in selected],fmt="o",color=COLORS["blue"],capsize=3)
        ax.set_yticks(y,[label for _,label in selected]);ax.invert_yaxis()
        ax.set_xlabel("RMSE" if metric=="rmse" else "NLL (nats per rating)")
        ax.set_title(title,fontsize=10);ax.grid(axis="x",alpha=.2)
    fig.tight_layout(w_pad=2)
    save(fig,figures/"baseline_comparison")

    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for k,c in [(2,COLORS["orange"]),(8,COLORS["blue"])]:
        t=trace[(trace.method=="ep")&(trace.k==k)].groupby("pass_index")
        means=t["validation_rmse"].mean();sd=t["validation_rmse"].std()
        axes[0].errorbar(means.index,means,yerr=sd,marker="o",capsize=3,label=f"K={k}",color=c)
        s=summary[(summary.method=="ep")&(summary.k==k)].sort_values("passes")
        axes[1].errorbar(s.train_seconds_mean,s.rmse_mean,yerr=s.rmse_std,xerr=s.train_seconds_std,
                         marker="o",capsize=3,label=f"K={k}",color=c)
        for _,r in s.iterrows():
            axes[1].annotate(str(int(r.passes)),(r.train_seconds_mean,r.rmse_mean),xytext=(4,4),textcoords="offset points",fontsize=8)
    axes[0].set(xlabel="EP passes",ylabel="Validation RMSE",title="Early iteration behaviour",xticks=range(1,6))
    axes[1].set(xlabel="Cumulative training time (seconds)",ylabel="Test RMSE",title="Accuracy versus computation")
    for ax in axes:ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,figures/"ep_tradeoff")

    bins=pd.read_csv(results/"calibration_bins.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for method,k,passes,label,color in [("mf",8,25,"MF",COLORS["green"]),("adf",8,1,"ADF",COLORS["orange"]),("ep",8,5,"EP5",COLORS["blue"])]:
        b=bins[(bins.method==method)&(bins.k==k)&(bins.passes==passes)].copy()
        b["weighted_conf"]=b.confidence*b["count"];b["weighted_acc"]=b.accuracy*b["count"]
        b=b.groupby("bin")[["count","weighted_conf","weighted_acc"]].sum()
        b=b[b["count"]>50]
        axes[0].plot(b.weighted_conf/b["count"],b.weighted_acc/b["count"],"o-",label=label,color=color)
        s=summary[(summary.method==method)&(summary.k==k)&(summary.passes==passes)].iloc[0]
        axes[1].bar(label,s.ece_mean,yerr=s.ece_std,capsize=4,color=color,width=.6)
    axes[0].plot([.2,.9],[.2,.9],"--",color="gray",lw=1)
    axes[0].set(xlabel="Mean top-class probability",ylabel="Observed top-class accuracy",title="Reliability (pooled descriptive bins)")
    axes[0].legend(frameon=False);axes[0].grid(alpha=.15)
    axes[1].set(ylabel="10-bin top-class ECE",title="Calibration did not improve with EP")
    fig.tight_layout();save(fig,figures/"calibration")

    table=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Method & MAE $\downarrow$ & RMSE $\downarrow$ & NLL $\downarrow$ & Fit time (s) \\",r"\midrule"]
    for row,label in selected:
        vals=[f"${row[c+'_mean']:.4f}\\pm{row[c+'_std']:.4f}$" for c in ["mae","rmse","nll"]]
        vals.append(f"${row.train_seconds_mean:.3f}\\pm{row.train_seconds_std:.3f}$")
        table.append(label+" & "+" & ".join(vals)+r" \\")
    raw=metrics[metrics.method=="mf"]
    vals=[f"${raw[c].mean():.4f}\\pm{raw[c].std():.4f}$" for c in ["raw_point_mae","raw_point_rmse"]]
    table.extend([r"\midrule",r"MF, clipped point & "+" & ".join(vals)+r" & same model & same model \\",r"\bottomrule",r"\end{tabular}"])
    (tables/"main_table.tex").write_text("\n".join(table)+"\n")
    table=[r"\begin{tabular}{llrrrr}",r"\toprule",r"$K$ & Schedule & RMSE & NLL & ECE & Site memory (MiB) \\",r"\midrule"]
    for k in [2,8]:
        for method,passes,label in [("adf",1,"ADF"),("ep",3,"EP3"),("ep",5,"EP5")]:
            row=summary[(summary.method==method)&(summary.k==k)&(summary.passes==passes)].iloc[0]
            table.append(f"{k} & {label} & {row.rmse_mean:.4f} & {row.nll_mean:.4f} & {row.ece_mean:.4f} & {row.site_memory_bytes_mean/2**20:.2f}"+r" \\")
    table.extend([r"\bottomrule",r"\end{tabular}"])
    (tables/"dimensions_table.tex").write_text("\n".join(table)+"\n")
    table=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Quantity & Quadrature & Hybrid & Absolute error & Reference change \\",r"\midrule"]
    b=pd.read_csv(results/"bilinear_validation.csv")
    names=[r"$\mathbb E[s]$",r"$\mathrm{Var}(s)$",r"$\mathbb E[t]$",r"$\mathrm{Var}(t)$",r"$\mathrm{Cov}(s,t)$"]
    for (_,row),name in zip(b.iterrows(),names):
        table.append(f"{name} & {row.quadrature80:.6f} & {row.hybrid:.6f} & {row.absolute_error:.6f} & {row.quadrature_change:.1e}"+r" \\")
    table.extend([r"\bottomrule",r"\end{tabular}"])
    (tables/"validation_table.tex").write_text("\n".join(table)+"\n")
    # Paired descriptive differences; three overlapping random splits do not
    # justify treating these as independent samples for a significance claim.
    pairs=[]
    for seed in sorted(metrics.seed.unique()):
        d=metrics[metrics.seed==seed]
        ep=d[(d.method=="ep")&(d.k==8)&(d.passes==5)].iloc[0]
        adf=d[(d.method=="adf")&(d.k==8)].iloc[0]
        mf=d[d.method=="mf"].iloc[0]
        pairs.append(dict(seed=int(seed),ep5_minus_adf_rmse=ep.rmse-adf.rmse,
                          ep5_minus_mf_distribution_rmse=ep.rmse-mf.rmse,
                          ep5_minus_mf_clipped_rmse=ep.rmse-mf.raw_point_rmse,
                          ep5_minus_mf_nll=ep.nll-mf.nll))
    pd.DataFrame(pairs).to_csv(results/"paired_differences.csv",index=False)
    local_figures=latex/"figures"
    local_figures.mkdir(exist_ok=True)
    for path in figures.glob("*.pdf"):
        shutil.copy2(path,local_figures/path.name)
    print("Generated figures, tables, graph and numerical validation evidence.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=ROOT/"Results/reference")
    parser.add_argument("--latex",type=Path,default=ROOT/"Report/latex")
    args=parser.parse_args();run(args.results,args.latex)
