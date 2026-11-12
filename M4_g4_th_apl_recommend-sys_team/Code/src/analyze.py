"""Recompute all summaries, conditional user-bootstrap intervals and figures."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from pathlib import Path
import json
import io
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=['#28536b','#de7c35','#387d67','#8b5d91','#c44e52','#6e7f91','#b69b28']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','axes.labelcolor':'#203040','figure.dpi':140,
                     'savefig.dpi':180,'font.family':'DejaVu Sans'})


def bootstrap(a,b,rng,nrep=2000):
    mask=a['count']>0
    count=a['count'][mask]
    assert np.array_equal(a['count'],b['count'])
    sa,sb=a['sse'][mask],b['sse'][mask]
    da,db=a['nll'][mask],b['nll'][mask]
    out=[]
    for start in range(0,nrep,100):
        idx=rng.integers(0,len(count),(min(100,nrep-start),len(count)))
        n=count[idx].sum(axis=1)
        d=np.sqrt(sa[idx].sum(axis=1)/n)-np.sqrt(sb[idx].sum(axis=1)/n)
        l=(da[idx].sum(axis=1)-db[idx].sum(axis=1))/n
        out.extend(zip(d,l))
    z=np.array(out)
    return np.quantile(z,[.025,.975],axis=0)


def savefig(fig,folder,name):
    for ext in ['png','pdf']:
        buffer=io.BytesIO()
        fig.savefig(buffer,format=ext,bbox_inches='tight',facecolor='white')
        payload=buffer.getvalue()
        target=folder/(name+'.'+ext)
        tmp=folder/(name+'.'+ext+'.tmp')
        tmp.write_bytes(payload)
        tmp.replace(target)
    plt.close(fig)


def load_many(root,name):
    files=[root/p/name for p in ['main','robustness','robustness_remaining','sensitivity'] if (root/p/name).exists()]
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True) if files else pd.DataFrame()


def latex_table(path,headers,rows,align=None):
    def esc(x):
        return re.sub(r'(?<!\\)%', r'\\%', str(x).replace('_',r'\_'))
    align=align or 'l'+'r'*(len(headers)-1)
    s=[r'\begin{tabular}{@{}'+align+r'@{}}\toprule',
       ' & '.join(esc(x) for x in headers)+r'\\\midrule']
    s+=[' & '.join(esc(x) for x in row)+r'\\' for row in rows]
    s += [r'\bottomrule\end{tabular}']
    path.write_text('\n'.join(s)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path('../Results'))
    ap.add_argument('--report',type=Path,default=Path('../Report/latex'))
    args=ap.parse_args();root=args.results
    dst=root/'analysis';figs=dst/'plots';figs.mkdir(parents=True,exist_ok=True)
    gen=args.report/'generated';gen.mkdir(parents=True,exist_ok=True)
    df=load_many(root,'metrics.csv');df.to_csv(dst/'all_metrics.csv',index=False)
    main_df=df[df.scenario=='main']
    metrics=['rmse','mae','nll','brier','ece','star_accuracy','fit_seconds','predict_seconds','clipped_rmse']
    metrics=[x for x in metrics if x in df]
    summary=df.groupby(['scenario','method'])[metrics].agg(['mean','std','count'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(dst/'summary.csv',index=False)
    sm=summary.loc['main']
    methods=['Global','UserMean','ItemMean','MF_SGD','ADF','EP3','EP5','EP10','MF_MAP','GMRF_MAP']
    labels={'Global':'Global','UserMean':'User mean','ItemMean':'Item mean','MF_SGD':'MF-SGD',
            'ADF':'ADF / EP1','EP3':'EP3','EP5':'EP5','EP10':'EP10','MF_MAP':'MF-MAP (no graph)','GMRF_MAP':'GMRF-MAP'}
    def fm(m,k,d=4):return f"{sm.loc[m,k+'_mean']:.{d}f} $\\pm$ {sm.loc[m,k+'_std']:.{d}f}"
    latex_table(gen/'m3_main.tex',['Method','RMSE','MAE','NLL'],[[labels[m]]+[fm(m,k) for k in ['rmse','mae','nll']] for m in methods])
    latex_table(gen/'m3_quality.tex',['Method','Brier','ECE','Star accuracy'],[[labels[m],fm(m,'brier'),fm(m,'ece'),f"{100*sm.loc[m,'star_accuracy_mean']:.2f} $\\pm$ {100*sm.loc[m,'star_accuracy_std']:.2f}\\%"] for m in methods[3:]])
    latex_table(gen/'m3_cost.tex',['Method','Fit (s)','Predict ($\\mu$s/rating)','Site MiB'],[[labels[m],fm(m,'fit_seconds',2),f"{1e6*sm.loc[m,'predict_seconds_mean']/100022:.3f}",'219.77' if m.startswith('EP') else '0'] for m in methods[3:]])
    # Paired conditional user bootstrap; split-specific intervals, no independence claim across splits.
    pairs=[('ADF','EP5'),('MF_MAP','GMRF_MAP'),('EP5','GMRF_MAP'),('EP10','GMRF_MAP')]
    records=[]
    for seed in sorted(main_df.seed.unique()):
        for a,b in pairs:
            pa=root/'main/user_stats'/f'{seed}_{a}.npz';pb=root/'main/user_stats'/f'{seed}_{b}.npz'
            if not (pa.exists() and pb.exists()):continue
            aa,bb=np.load(pa),np.load(pb)
            ci=bootstrap(aa,bb,np.random.default_rng(int(seed)+12000),2000)
            n=aa['count'].sum()
            records.append(dict(seed=seed,baseline=a,comparison=b,
                rmse_reduction=float(np.sqrt(aa['sse'].sum()/n)-np.sqrt(bb['sse'].sum()/n)),
                rmse_ci_low=float(ci[0,0]),rmse_ci_high=float(ci[1,0]),
                nll_reduction=float((aa['nll'].sum()-bb['nll'].sum())/n),
                nll_ci_low=float(ci[0,1]),nll_ci_high=float(ci[1,1]),replicates=2000))
    ci=pd.DataFrame(records);ci.to_csv(dst/'paired_user_bootstrap.csv',index=False)
    graph_ci=ci[(ci.baseline=='MF_MAP')&(ci.comparison=='GMRF_MAP')]
    latex_table(gen/'m3_bootstrap.tex',['Seed','RMSE reduction','95\\% user-bootstrap interval'],[[int(r.seed),f'{r.rmse_reduction:+.6f}',f'[{r.rmse_ci_low:+.6f}, {r.rmse_ci_high:+.6f}]'] for r in graph_ci.itertuples()])
    grid=pd.read_csv(root/'main/validation_grid.csv')
    grid=grid[grid.model=='MAP']
    fig,ax=plt.subplots(1,2,figsize=(10.6,4.1))
    vg=grid.groupby('gamma').validation_rmse.agg(['mean','std'])
    ax[0].errorbar(np.arange(len(vg)),vg['mean'],yerr=vg['std'],fmt='o-',capsize=4,color=COLORS[0])
    ax[0].set_xticks(np.arange(len(vg)),[f'{g:g}' for g in vg.index]);ax[0].set(xlabel='Graph strength gamma',ylabel='Validation clipped-score RMSE',title='Validation-only graph-strength search')
    yy=np.arange(len(graph_ci));gg=graph_ci.sort_values('seed')
    ax[1].errorbar(gg.rmse_reduction,yy,xerr=np.array([gg.rmse_reduction-gg.rmse_ci_low,gg.rmse_ci_high-gg.rmse_reduction]),fmt='o',capsize=4,color=COLORS[2])
    ax[1].axvline(0,ls='--',color='gray');ax[1].set_yticks(yy,[str(x) for x in gg.seed]);ax[1].set(xlabel='RMSE reduction from the graph',ylabel='Split seed',title='Paired user-bootstrap 95% intervals')
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout();savefig(fig,figs,'graph_evidence')
    # Main error/probability comparison.
    pick=methods[3:];x=np.arange(len(pick))
    fig,ax=plt.subplots(1,2,figsize=(11,4.1))
    for a,key,title in zip(ax,['rmse','nll'],['Rating prediction','Probability quality']):
        
        for j,m in enumerate(pick):
            a.errorbar(j,sm.loc[m,key+'_mean'],yerr=sm.loc[m,key+'_std'],fmt='o',markersize=7,color=COLORS[j%len(COLORS)],capsize=4)
        a.set_xticks(x,[labels[m].replace(' (no graph)','') for m in pick],rotation=35,ha='right')
        a.set_ylabel(key.upper()+' (lower is better)');a.set_title(title)
        a.set_ylim((.82,.92) if key=='rmse' else (1.17,1.28));a.grid(axis='y',alpha=.2)
    fig.suptitle('MovieLens 1M: mean and sample SD across five splits',y=1.03)
    fig.tight_layout();savefig(fig,figs,'main_comparison')
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    for j,m in enumerate(pick):
        a=sm.loc[m];ax.errorbar(a.fit_seconds_mean,a.rmse_mean,xerr=a.fit_seconds_std,yerr=a.rmse_std,fmt='o',capsize=3,color=COLORS[j%len(COLORS)])
        ax.annotate(labels[m].replace(' (no graph)',''),(a.fit_seconds_mean,a.rmse_mean),xytext=(5,5 if m!='GMRF_MAP' else -13),textcoords='offset points',fontsize=9)
    ax.set_xscale('log');ax.set_xlabel('Fit seconds, log scale (selected candidate only)');ax.set_ylabel('Test expected-rating RMSE');ax.grid(alpha=.2)
    ax.set_title('Accuracy versus fitting cost; graph/search overhead reported separately')
    savefig(fig,figs,'cost_tradeoff')
    # Reliability curves: counts pooled descriptively, not independent observations across seeds.
    cb=load_many(root,'calibration_bins.csv');cb=cb[cb.scenario=='main']
    fig,ax=plt.subplots(figsize=(7.2,4.6));ax.plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
    for j,m in enumerate(['ADF','EP5','EP10','GMRF_MAP']):
        z=cb[cb.method==m].groupby('bin')[['n','confidence_sum','correct']].sum();z=z[z.n>50]
        ax.plot(z.confidence_sum/z.n,z.correct/z.n,'o-',label=labels[m],color=COLORS[j])
    ax.set(xlabel='Mean top-class probability',ylabel='Observed top-class accuracy',xlim=(.2,1),ylim=(.2,1),title='Reliability diagram (pooled bins are descriptive)');ax.legend();ax.grid(alpha=.2)
    savefig(fig,figs,'reliability')
    # Validation convergence; no test-based stopping.
    tr=load_many(root,'trace.csv');tr=tr[tr.scenario=='main']
    fig,ax=plt.subplots(1,2,figsize=(10.6,4))
    ep=tr[tr.method=='EP'].groupby('epoch').validation_rmse.agg(['mean','std'])
    ax[0].errorbar(ep.index,ep['mean'],yerr=ep['std'],marker='o',capsize=3,color=COLORS[0]);ax[0].set(xlabel='EP pass',ylabel='Validation expected-rating RMSE',title='EP passes (mean and SD)')
    for j,g in enumerate([0.,10.]):
        z=tr[(tr.method=='MAP')&(tr.gamma==g)].copy()
        z['relative_objective']=z.objective/z.groupby('seed').objective.transform('first')
        s=z.groupby('epoch').relative_objective.mean()
        ax[1].plot(s.index,s.values,label=f'gamma={g:g}',color=COLORS[j])
    ax[1].set(xlabel='MAP epoch',ylabel='Objective / epoch-1 objective',title='Block minimization decreases the objective');ax[1].legend()
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout();savefig(fig,figs,'convergence')
    # Matched robustness curves use seeds 42--44, including their clean endpoints.
    robust=df[df.seed.isin([42,43,44])]
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.2))
    for panel,cases,xs,title,xlabel in [(0,['retain_0.25','retain_0.5','main'],[25,50,100],'Missing training ratings','Original training ratings retained (%)'),
                                      (1,['main','noise_0.1','noise_0.2'],[0,10,20],'Corrupted training labels','Training ratings replaced (%)')]:
        for j,m in enumerate(['ADF','EP5','MF_MAP','GMRF_MAP']):
            z=robust[(robust.method==m)&robust.scenario.isin(cases)].groupby('scenario').rmse.agg(['mean','std'])
            if all(c in z.index for c in cases):
                ax[panel].errorbar(xs,z.loc[cases,'mean'],yerr=z.loc[cases,'std'],marker='o',capsize=3,label=labels[m],color=COLORS[j])
        ax[panel].set(xlabel=xlabel,ylabel='Clean-test expected-rating RMSE',title=title);ax[panel].grid(alpha=.2)
    ax[0].legend(fontsize=8);fig.tight_layout();savefig(fig,figs,'robustness')
    # Graph ablation table / plot.
    cases=['main','ratings_only','genres_only','graph_top_5','graph_top_50','cold_items']
    rows=[]
    for c in cases:
        for m in ['MF_MAP','GMRF_MAP']:
            z=robust[(robust.scenario==c)&(robust.method==m)]
            if len(z):rows.append(dict(scenario=c,method=m,n=len(z),rmse_mean=z.rmse.mean(),rmse_std=z.rmse.std(),nll_mean=z.nll.mean(),nll_std=z.nll.std()))
    ab=pd.DataFrame(rows);ab.to_csv(dst/'graph_ablation.csv',index=False)
    latex_table(gen/'m3_robustness.tex',['Condition','ADF','EP5','MF-MAP','GMRF-MAP'],[
        [c]+[f"{z.rmse.mean():.4f} $\\pm$ {z.rmse.std():.4f}" if len(z) else '--' for m in ['ADF','EP5','MF_MAP','GMRF_MAP'] for z in [robust[(robust.scenario==c)&(robust.method==m)]]]
        for c in ['main','retain_0.25','retain_0.5','noise_0.1','noise_0.2','cold_items','cold_users']])
    latex_table(gen/'m3_ablation.tex',['Graph condition','MF-MAP RMSE','GMRF-MAP RMSE'],[
        [c]+[f"{z.rmse.mean():.4f} $\\pm$ {z.rmse.std():.4f}" if len(z) else '--' for m in ['MF_MAP','GMRF_MAP'] for z in [robust[(robust.scenario==c)&(robust.method==m)]]]
        for c in cases[:-1]])
    # Per-class and item-support errors.
    eg=load_many(root,'error_groups.csv');eg=eg[(eg.scenario=='main')&eg.method.isin(['ADF','EP5','EP10','GMRF_MAP'])]
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.2))
    for panel,group,title in [(0,'actual_rating','Error by true rating'),(1,'item_support','Error by item training support')]:
        for j,m in enumerate(['ADF','EP5','EP10','GMRF_MAP']):
            z=eg[(eg.method==m)&(eg.group==group)].groupby('value')[['n','sse']].sum()
            ax[panel].plot(z.index,np.sqrt(z.sse/z.n),'o-',label=labels[m],color=COLORS[j])
        ax[panel].set(ylabel='Expected-rating RMSE',title=title);ax[panel].grid(alpha=.2)
    ax[0].set_xlabel('True star rating');ax[0].set_xticks([1,2,3,4,5]);ax[0].legend(fontsize=8)
    ax[1].set_xticks([0,1,2,3,4],['0','1-19','20-99','100-499','500+']);ax[1].set_xlabel('Ratings per movie in training')
    fig.tight_layout();savefig(fig,figs,'error_analysis')
    # Noise variance and numerical diagnostics.
    ns=['EP5_noise_0.5','EP5','EP5_noise_2']
    latex_table(gen/'m3_noise.tex',['Noise variance','RMSE','NLL','ECE'],[[v]+[fm(m,k) for k in ['rmse','nll','ece']] for v,m in zip(['0.5','1.0','2.0'],ns)])
    sens=df[df.scenario=='sensitivity']
    if len(sens):
        ss=sens.groupby('method')[['rmse','nll']].agg(['mean','std'])
        latex_table(gen/'m3_sensitivity.tex',['EP5 variant','RMSE','NLL'],[[m.replace('EP5_','')]+[f'{ss.loc[m,(k,"mean")]:.4f} $\\pm$ {ss.loc[m,(k,"std")]:.4f}' for k in ['rmse','nll']] for m in ss.index])
    if (root/'validation/gaussian_inference.csv').exists():
        q=pd.read_csv(root/'validation/gaussian_inference.csv');s=q.groupby(['method','budget'])[['mean_rmse','variance_rmse','seconds']].agg(['mean','std'])
        s.to_csv(dst/'gaussian_inference_summary.csv')
        latex_table(gen/'m3_inference.tex',['Method','Budget','Mean RMSE','Variance RMSE','Time (ms)'],[
            [m,int(n),f'{r[("mean_rmse","mean")]:.3g}',f'{r[("variance_rmse","mean")]:.3g}' if np.isfinite(r[("variance_rmse","mean")]) else '--',f'{1000*r[("seconds","mean")]:.3f}'] for (m,n),r in s.iterrows()])
    # Actual four-movie subgraph (seed 42), with all edges among selected nodes.
    graphfile=root/'main/graphs/graph_42.npz'
    if graphfile.exists():
        from scipy import sparse
        w=sparse.load_npz(graphfile)
        movie_ids=np.load(root/'main/id_mapping.npz')['movie_ids']
        chosen=[1,3114,2355,3751]
        names=['Toy Story (1)','Toy Story 2 (3114)',"A Bug's Life (2355)",'Chicken Run (3751)']
        ids=[int(np.flatnonzero(movie_ids==i)[0]) for i in chosen]
        xy=np.array([[0,1.1],[1.4,0],[0,-1.1],[-1.4,0]])
        fig,ax=plt.subplots(figsize=(8.1,4.5))
        for i in range(4):
            for j in range(i+1,4):
                v=float(w[ids[i],ids[j]])
                if v>0:
                    ax.plot(xy[[i,j],0],xy[[i,j],1],color='#78909c',linewidth=1+20*v,zorder=1)
                    mid=(xy[i]+xy[j])/2
                    if (i,j)==(0,2):mid=mid+np.array([.15,.2])
                    if (i,j)==(1,3):mid=mid+np.array([-.1,-.2])
                    ax.text(mid[0],mid[1],f'{v:.3f}',ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=1.5),zorder=3)
        for (x,y),name in zip(xy,names):
            ax.text(x,y,name+'\nmovie trait vector',ha='center',va='center',fontsize=10,color='white',bbox=dict(boxstyle='round,pad=.5',facecolor='#28536b',edgecolor='white'),zorder=4)
        ax.set(xlim=(-2.4,2.4),ylim=(-1.6,1.7),title='Four movies from the constructed Markov graph')
        ax.axis('off');savefig(fig,figs,'movie_subgraph')
    # Compact machine-readable evidence used to write conclusions.
    evidence=dict(main=main_df.groupby('method')[metrics].mean().to_dict('index'),
                  graph_ci=graph_ci.to_dict('records'),
                  total_metric_rows=len(df),main_seeds=sorted(int(x) for x in main_df.seed.unique()),
                  robustness_conditions=sorted(str(x) for x in robust.scenario.unique()))
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    (dst/'evidence.json').write_text(json.dumps(clean(evidence),indent=2,allow_nan=False)+'\n')
    print('Analysis written to',dst.resolve())


if __name__=='__main__':main()
