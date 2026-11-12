"""Audit completed experiment coverage, selections and numerical invariants."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);args=ap.parse_args()
    root=args.results
    files=[root/x/'metrics.csv' for x in ['main','robustness','robustness_remaining','sensitivity'] if (root/x/'metrics.csv').exists()]
    df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    assert not df.duplicated(['scenario','seed','method']).any(),'Duplicate comparison rows'
    assert np.isfinite(df[['rmse','mae','nll','brier','ece']].to_numpy()).all()
    main=df[df.scenario=='main']
    assert set(main.seed)=={42,43,44,45,46}
    assert main.groupby('seed').size().eq(17).all()
    cases=['retain_0.25','retain_0.5','noise_0.1','noise_0.2','graph_top_5','graph_top_50','ratings_only','genres_only','cold_items','cold_users']
    for seed in [42,43,44]:
        for case in cases:
            q=df[(df.scenario==case)&(df.seed==seed)]
            needed={'MF_MAP','GMRF_MAP','Graph_selected'}
            if case.startswith(('retain_','noise_','cold_')):needed|={'ADF','EP5'}
            assert set(q.method)==needed,(seed,case,'incomplete')
        q=df[(df.scenario=='sensitivity')&(df.seed==seed)]
        assert len(q)==6,(seed,'incomplete sensitivity')
    for stage in ['main','robustness','robustness_remaining']:
        p=root/stage
        if not p.exists():continue
        grid=pd.read_csv(p/'validation_grid.csv')
        rows=pd.read_csv(p/'metrics.csv')
        for (case,seed),q in grid[grid.model=='MAP'].groupby(['scenario','seed']):
            chosen=q.sort_values(['validation_rmse','gamma']).iloc[0]
            r=rows[(rows.scenario==case)&(rows.seed==seed)&(rows.method=='Graph_selected')].iloc[0]
            assert r.gamma==chosen.gamma and r.epoch==chosen.selected_epoch
            g=q[q.gamma>0].sort_values(['validation_rmse','gamma']).iloc[0]
            r=rows[(rows.scenario==case)&(rows.seed==seed)&(rows.method=='GMRF_MAP')].iloc[0]
            assert r.gamma==g.gamma and r.epoch==g.selected_epoch
        trace=pd.read_csv(p/'trace.csv')
        for _,q in trace[trace.method=='MAP'].groupby(['scenario','seed','gamma']):
            e=q.sort_values('epoch').objective.to_numpy()
            assert np.max(np.diff(e))<=1e-7,'MAP objective increased'
    for seed in [42,43,44,45,46]:
        q=main[main.seed==seed].set_index('method')
        for metric in ['rmse','mae','nll','brier','ece']:
            assert q.loc['ADF',metric]==q.loc['EP1',metric]
        g=pd.read_csv(root/'main/validation_grid.csv')
        z=g[(g.seed==seed)&(g.model=='EP_noise')]
        chosen=z.sort_values('validation_nll').iloc[0]
        assert q.loc['EP_noise_tuned','noise_variance']==chosen.noise_variance
    # Main numerical source bytes are the exact executed snapshot.
    hashes=json.loads((root/'main/source_hashes.json').read_text())
    for name,value in hashes.items():
        p=root/'main/source_snapshot'/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==value,(name,'source mismatch')
    result=dict(passed=True,metric_rows=len(df),main_rows=len(main),main_seeds=5,
        robustness_conditions_per_seed=10,robustness_seeds=3,sensitivity_rows=18,
        checks=['complete coverage','no duplicate rows','finite KPIs','validation-only recorded selection',
        'MAP objective nonincrease','ADF/EP1 equality','noise selection','main source hashes'])
    (root/'analysis/result_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
