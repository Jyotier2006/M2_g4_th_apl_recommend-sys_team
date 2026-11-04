"""M3 experiments: procedural functions, arrays and explicit loops (CP style).

All models share row splits. Only training inputs build graphs or fit parameters.
Validation chooses checkpoints/hyperparameters; test results never select a model.
The main and robustness protocols are fixed in configs/m3.json before execution.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'

import numpy as np
import scipy
import numba
from scipy import sparse
import bayes
import gmrf
from sgd_baseline import sgd_epoch
from data import load_dataset, split_indices, split_audit, write_json
from metrics import evaluate


def csvwrite(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def copy_model(m):
    return {k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in m.items() if k not in ['sp', 'sh']}


def warm():
    u = np.array([0, 0, 1, 1], dtype=np.int64)
    i = np.array([0, 1, 0, 1], dtype=np.int64)
    r = np.array([2, 3, 4, 5], dtype=np.int64)
    for mode in ['adf', 'ep']:
        m = bayes.init(2, 2, 2, 42, r, mode)
        bayes.step(m, u, i, r, np.arange(4))
    w = sparse.csr_matrix([[0., 1.], [1., 0.]])
    gmrf.fit(u, i, r, u, i, r, 2, 2, 2, 42, w, 1., epochs=1)
    m = gmrf.init(2, 2, 2, 42, r)
    sgd_epoch(u, i, r, np.arange(4), m['p'], m['q'], m['bu'], m['bi'], m['mu'], .007, .03)


def run_ep(u, i, r, nu, ni, seed, k, passes, vu, vi, vr,
           noise=1., prior_var=.25, damping=.5, shuffle=False):
    t0 = time.perf_counter()
    m = bayes.init(nu, ni, k, seed, r, 'ep', noise, prior_var)
    seconds = time.perf_counter()-t0
    order = np.arange(len(r))
    states, trace = {}, []
    rng = np.random.default_rng(seed+7000)
    for ep in range(1, passes+1):
        if shuffle:
            order = rng.permutation(len(r))
        old = np.r_[m['uh'].ravel()/m['up'].ravel(), m['ih'].ravel()/m['ip'].ravel()]
        t0 = time.perf_counter()
        diag = bayes.step(m, u, i, r, order, outer_damping=damping)
        seconds += time.perf_counter()-t0
        new = np.r_[m['uh'].ravel()/m['up'].ravel(), m['ih'].ravel()/m['ip'].ravel()]
        val = evaluate(vr, bayes.prob(m, vu, vi))
        trace.append(dict(epoch=ep, fit_seconds=seconds, noise_variance=noise,
                          validation_rmse=val['rmse'], validation_nll=val['nll'],
                          parameter_rms_change=float(np.sqrt(np.mean((new-old)**2))), **diag))
        if ep in {1, 2, 3, 4, 5, passes}:
            states[ep] = (copy_model(m), seconds)
    return states, trace


def run_sgd(u, i, r, nu, ni, seed, k, vu, vi, vr, epochs=25):
    m = gmrf.init(nu, ni, k, seed, r)
    order = np.arange(len(r))
    best, state, trace = np.inf, None, []
    start = time.perf_counter()
    for ep in range(epochs):
        sgd_epoch(u, i, r, order, m['p'], m['q'], m['bu'], m['bi'], m['mu'], .007*.98**ep, .03)
        rmse = float(np.sqrt(np.mean((np.clip(gmrf.point(m, vu, vi), 1, 5)-vr)**2)))
        trace.append(dict(epoch=ep+1, validation_rmse=rmse))
        if rmse < best:
            best, state = rmse, copy_model(m)
            state['epoch'] = ep+1
    gmrf.calibrate(state, vu, vi, vr)
    return state, trace, time.perf_counter()-start


def mean_prob(u, i, r, tu, ti, nu, ni, kind):
    g = np.bincount(r, minlength=6)[1:].astype(float)
    g /= g.sum()
    if kind == 'Global':
        return np.tile(g, (len(tu), 1))
    ids, n, test = (u, nu, tu) if kind == 'UserMean' else (i, ni, ti)
    a = np.tile(10*g, (n, 1))
    np.add.at(a, (ids, r-1), 1)
    return a[test]/a[test].sum(axis=1, keepdims=True)


def record(out, rows, bins, groups, ctx, name, pr, seconds, pp=None, extra=None):
    r, u, i = ctx['test_r'], ctx['test_u'], ctx['test_i']
    pred = pr@np.arange(1, 6)
    ans = evaluate(r, pr)
    ans['star_accuracy'] = float(np.mean(pr.argmax(axis=1)+1 == r))
    ans['within_one'] = float(np.mean(np.abs(pred-r) <= 1))
    if pp is not None:
        ans['clipped_rmse'] = float(np.sqrt(np.mean((np.clip(pp, 1, 5)-r)**2)))
    row = dict(scenario=ctx['scenario'], seed=ctx['seed'], method=name,
               train_n=len(ctx['train_r']), test_n=len(r), fit_seconds=seconds, **ans, **(extra or {}))
    rows.append(row)
    csvwrite(out/'metrics.csv', rows)
    conf, hit = pr.max(axis=1), (pr.argmax(axis=1)+1 == r)
    for b in range(10):
        mask = (conf >= b/10) & ((conf < (b+1)/10) if b < 9 else (conf <= 1))
        bins.append(dict(scenario=ctx['scenario'], seed=ctx['seed'], method=name, bin=b,
                         n=int(mask.sum()), confidence_sum=float(conf[mask].sum()), correct=int(hit[mask].sum())))
    for label, cat in [('actual_rating', r),
                       ('item_support', np.digitize(ctx['item_counts'][i], [1, 20, 100, 500])),
                       ('user_support', np.digitize(ctx['user_counts'][u], [1, 20, 100, 500]))]:
        for value in np.unique(cat):
            mask = cat == value
            groups.append(dict(scenario=ctx['scenario'], seed=ctx['seed'], method=name,
                group=label, value=int(value), n=int(mask.sum()),
                sse=float(np.sum((pred[mask]-r[mask])**2)),
                abs_error=float(np.abs(pred[mask]-r[mask]).sum()),
                nll_sum=float(-np.log(pr[mask, r[mask]-1]).sum()),
                signed_error=float((pred[mask]-r[mask]).sum())))
    # User-cluster sums suffice to reproduce paired bootstrap CIs without large prediction dumps.
    if ctx['scenario'] == 'main':
        stats = dict(count=np.bincount(u, minlength=ctx['nu']),
                     sse=np.bincount(u, weights=(pred-r)**2, minlength=ctx['nu']),
                     nll=np.bincount(u, weights=-np.log(pr[np.arange(len(r)), r-1]), minlength=ctx['nu']))
        if pp is not None:
            stats['clipped_sse'] = np.bincount(u, weights=(np.clip(pp,1,5)-r)**2, minlength=ctx['nu'])
        np.savez_compressed(out/'user_stats'/f"{ctx['seed']}_{name}.npz", **stats)
    print(f"{ctx['scenario']} seed={ctx['seed']} {name}: RMSE={ans['rmse']:.6f}, NLL={ans['nll']:.6f}, fit={seconds:.2f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('../Data/ml-1m'))
    ap.add_argument('--out', type=Path, default=Path('../Results/m3'))
    ap.add_argument('--config', type=Path, default=Path(__file__).parents[1]/'configs/m3.json')
    ap.add_argument('--conditions', nargs='+', help='Run only these named conditions (for isolated reruns/recovery).')
    ap.add_argument('--stage', choices=['main','robustness','sensitivity','all'], default='all')
    ap.add_argument('--seeds', type=int, nargs='+', help='Optional explicit seed override; recorded in config.')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    seeds = args.seeds or (cfg['seeds'] if args.stage in ['main','all'] else cfg['robustness_seeds'])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists():
        raise FileExistsError('Choose a fresh --out directory; completed/partial runs are never overwritten.')
    for sub in ['user_stats','models','graphs']:
        (out/sub).mkdir()
    cfg.update(stage=args.stage, executed_seeds=seeds, data_path=str(args.data.resolve()), condition_filter=args.conditions)
    write_json(out/'config.json', cfg)
    write_json(out/'source_hashes.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})
    write_json(out/'environment.json', dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        numba=numba.__version__, platform=platform.platform(), cpu=platform.processor(), threads=1,
        timing='Sequential CPU; JIT warmup excluded. MAP/SGD fitting includes validation checks and calibration. EP validation scoring excluded. Graph construction and total hyperparameter search reported separately.'))
    print('Warming kernels; compilation excluded from fit timing.', flush=True)
    warm()
    data = load_dataset(args.data)
    nu, ni = len(data['uid']), len(data['iid'])
    genres, titles, genre_names = gmrf.movie_info(args.data, data['iid'])
    data['stats']['metadata_used_for_training'] = {'EP':False, 'GMRF': 'movie genres only; no user demographics'}
    write_json(out/'dataset.json', data['stats'])
    np.savez_compressed(out/'id_mapping.npz', user_ids=data['uid'], movie_ids=data['iid'])
    rows, bins, groups, traces, grids, audits, graph_stats = [], [], [], [], [], [], []
    k = cfg['k']
    for seed in seeds:
        tr0, va, te = split_indices(len(data['r']), seed)
        audits.append(split_audit(data, (tr0, va, te), seed))
        write_json(out/'split_audits.json', audits)
        scenarios = []
        if args.stage in ['main','all']:
            scenarios += [('main', 1., 0., cfg['top'], cfg['genre_mix'])]
        if args.stage in ['robustness','all'] and (args.seeds or seed in cfg['robustness_seeds']):
            scenarios += [(f'retain_{f:g}', f, 0., cfg['top'], cfg['genre_mix']) for f in cfg['train_fractions']]
            scenarios += [(f'noise_{f:g}', 1., f, cfg['top'], cfg['genre_mix']) for f in cfg['corruption_fractions']]
            scenarios += [(f'graph_top_{n}', 1., 0., n, cfg['genre_mix']) for n in cfg['graph_neighbours']]
            scenarios += [('ratings_only', 1., 0., cfg['top'], 0.), ('genres_only', 1., 0., cfg['top'], 1.)]
            scenarios += [('cold_items', 1., 0., cfg['top'], cfg['genre_mix']), ('cold_users', 1., 0., cfg['top'], cfg['genre_mix'])]
        if args.stage == 'sensitivity':
            scenarios = [('sensitivity', 1., 0., cfg['top'], cfg['genre_mix'])]
        if args.conditions:
            scenarios=[x for x in scenarios if x[0] in args.conditions]
        for scenario, frac, corrupt, top, mix in scenarios:
            tr = tr0[:int(len(tr0)*frac)]
            local_te = te
            if scenario == 'cold_items':
                # Hold out 10% of item identities entirely from fitting; validation/test remain separate.
                cold = np.random.default_rng(seed+9100).choice(ni, size=int(.1*ni), replace=False)
                tr = tr[~np.isin(data['i'][tr], cold)]
                local_te = te[np.isin(data['i'][te], cold)]
            if scenario == 'cold_users':
                cold = np.random.default_rng(seed+9200).choice(nu, size=int(.1*nu), replace=False)
                tr = tr[~np.isin(data['u'][tr], cold)]
                local_te = te[np.isin(data['u'][te], cold)]
            u, i, r = data['u'][tr], data['i'][tr], data['r'][tr].copy()
            if corrupt:
                # Uniform replacement by a DIFFERENT star; exactly floor(fraction*N) labels changed.
                rng = np.random.default_rng(seed+8100)
                z = rng.permutation(len(r))[:int(corrupt*len(r))]
                r[z] = (r[z]-1+rng.integers(1,5,len(z)))%5+1
            write_json(out/f'audit_{seed}_{scenario}.json', dict(train_rows=len(tr), test_rows=len(local_te),
                train_row_hash=hashlib.sha256(tr.astype('<i8').tobytes()).hexdigest(),
                train_label_hash=hashlib.sha256(r.astype('<i8').tobytes()).hexdigest(),
                corrupted_labels=int(np.sum(r!=data['r'][tr])), retained_fraction=frac,
                noise_fraction=corrupt, top=top, genre_mix=mix,
                graph_inputs='condition training ratings and fixed genres only'))
            vu, vi, vr = data['u'][va], data['i'][va], data['r'][va]
            tu, ti, rr = data['u'][local_te], data['i'][local_te], data['r'][local_te]
            ctx = dict(scenario=scenario, seed=seed, test_u=tu, test_i=ti, test_r=rr,
                train_r=r, nu=nu, item_counts=np.bincount(i,minlength=ni), user_counts=np.bincount(u,minlength=nu))
            def rec(name, m, sec, kind='map', extra=None):
                t = time.perf_counter()
                pr = bayes.prob(m,tu,ti) if kind=='ep' else gmrf.prob(m,tu,ti)
                predsec = time.perf_counter()-t
                pp = None if kind=='ep' else gmrf.point(m,tu,ti)
                record(out,rows,bins,groups,ctx,name,pr,sec,pp,dict(predict_seconds=predsec,**(extra or {})))
                if scenario=='main' and seed==seeds[0] and name in ['EP5','EP10','EP_noise_tuned','MF_MAP','GMRF_MAP','MF_SGD']:
                    np.savez_compressed(out/'models'/f'{seed}_{name}.npz',**copy_model(m))
            if scenario == 'main':
                for name in ['Global','UserMean','ItemMean']:
                    t = time.perf_counter()
                    pr = mean_prob(u,i,r,tu,ti,nu,ni,name)
                    record(out,rows,bins,groups,ctx,name,pr,time.perf_counter()-t)
                m, tt, sec = run_sgd(u,i,r,nu,ni,seed,k,vu,vi,vr)
                rec('MF_SGD',m,sec,extra={'epoch':m['epoch'],'noise_variance':m['noise']})
                traces += [dict(scenario=scenario,seed=seed,method='MF_SGD',**x) for x in tt]
            ep_needed = scenario in ['main','cold_items','cold_users','sensitivity'] or scenario.startswith(('retain_','noise_'))
            if ep_needed and scenario!='sensitivity':
                t = time.perf_counter()
                ad = bayes.init(nu,ni,k,seed,r,'adf')
                diag = bayes.step(ad,u,i,r,np.arange(len(r)))
                sec = time.perf_counter()-t
                rec('ADF',ad,sec,'ep',diag)
                states, tt = run_ep(u,i,r,nu,ni,seed,k,cfg['ep_passes'] if scenario=='main' else 5,vu,vi,vr)
                for key in ['up','uh','ip','ih']:
                    np.testing.assert_array_equal(ad[key],states[1][0][key])
                del ad
                traces += [dict(scenario=scenario,seed=seed,method='EP',**x) for x in tt]
                for ep,(m,sec) in states.items():
                    if scenario=='main' or ep==5:
                        rec(f'EP{ep}',m,sec,'ep',dict(passes=ep,noise_variance=1.))
                if scenario=='main':
                    candidate = [(1., states[5][0], states[5][1], tt[4]['validation_nll'])]
                    for noise in cfg['noise_variances']:
                        if noise==1.:
                            continue
                        ss, nt = run_ep(u,i,r,nu,ni,seed,k,5,vu,vi,vr,noise=noise)
                        m,sec = ss[5]
                        rec(f'EP5_noise_{noise:g}',m,sec,'ep',dict(noise_variance=noise))
                        candidate.append((noise,m,sec,nt[-1]['validation_nll']))
                        traces += [dict(scenario=scenario,seed=seed,method=f'EP_noise_{noise:g}',**x) for x in nt]
                    noise,m,sec,vnll = min(candidate,key=lambda x:x[3])
                    rec('EP_noise_tuned',m,sec,'ep',dict(noise_variance=noise,validation_nll=vnll,
                        full_search_seconds=sum(x[2] for x in candidate)))
                    grids += [dict(scenario=scenario,seed=seed,model='EP_noise',noise_variance=x[0],validation_nll=x[3]) for x in candidate]
                del states
            if scenario=='sensitivity':
                for name, opts in [('prior_0.1',dict(prior_var=.1)),('prior_0.5',dict(prior_var=.5)),
                                   ('damping_0.25',dict(damping=.25)),('damping_1',dict(damping=1.)),
                                   ('reshuffled',dict(shuffle=True))]:
                    ss, tt = run_ep(u,i,r,nu,ni,seed,k,5,vu,vi,vr,**opts)
                    m,sec=ss[5]
                    rec('EP5_'+name,m,sec,'ep')
                    if name=='reshuffled':
                        rec('ADF_reshuffled',ss[1][0],ss[1][1],'ep')
                    traces += [dict(scenario=scenario,seed=seed,method='EP5_'+name,**x) for x in tt]
                continue
            t=time.perf_counter()
            w=gmrf.graph(u,i,r,nu,ni,genres,top=top,mix=mix)
            graph_sec=time.perf_counter()-t
            edge=sparse.triu(w,k=1).tocoo()
            gs=dict(scenario=scenario,seed=seed,movies=ni,edges=len(edge.data),
                density=2*len(edge.data)/(ni*(ni-1)),mean_degree=w.nnz/ni,
                isolated_movies=int((np.diff(w.indptr)==0).sum()),graph_seconds=graph_sec,top=top,genre_mix=mix,
                unary_trait_factors=ni*k,pairwise_trait_factors=len(edge.data)*k,scalar_trait_bias_variables=(nu+ni)*(k+1))
            graph_stats.append(gs)
            if scenario=='main' and seed==seeds[0]:
                sparse.save_npz(out/'graphs'/f'graph_{seed}.npz',w)
                csvwrite(out/'graphs'/f'edges_{seed}.csv', [dict(movie_a=int(data['iid'][a]),movie_b=int(data['iid'][b]),weight=float(v)) for a,b,v in zip(edge.row,edge.col,edge.data)])
            candidates=[]
            # The same gamma grid and epoch budget are used in every MAP condition.
            for gamma in cfg['gammas']:
                m,tt,sec=gmrf.fit(u,i,r,vu,vi,vr,nu,ni,k,seed,w,gamma,
                    cfg['map_epochs'],cfg['reg'],cfg['bias_reg'])
                candidates.append((gamma,m,sec))
                grids.append(dict(scenario=scenario,seed=seed,model='MAP',gamma=gamma,
                    selected_epoch=m['epoch'],validation_rmse=m['validation_rmse'],fit_seconds=sec))
                traces += [dict(scenario=scenario,seed=seed,method='MAP',gamma=gamma,**x) for x in tt]
                print(f'  {scenario} seed={seed} gamma={gamma:g}: validation RMSE={m["validation_rmse"]:.6f}, epoch={m["epoch"]}',flush=True)
            base=next(x for x in candidates if x[0]==0)
            graph=min((x for x in candidates if x[0]>0),key=lambda x:x[1]['validation_rmse'])
            adaptive=min(candidates,key=lambda x:x[1]['validation_rmse'])
            for name,(gamma,m,sec) in [('MF_MAP',base),('GMRF_MAP',graph),('Graph_selected',adaptive)]:
                rec(name,m,sec,extra=dict(gamma=gamma,epoch=m['epoch'],noise_variance=m['noise'],
                    validation_rmse=m['validation_rmse'],graph_seconds=graph_sec,
                    full_search_seconds=sum(x[2] for x in candidates),edges=len(edge.data)))
            for fn, rr0 in [('trace.csv',traces),('validation_grid.csv',grids),('calibration_bins.csv',bins),
                            ('error_groups.csv',groups),('graph_stats.csv',graph_stats)]:
                csvwrite(out/fn,rr0)
        for fn,rr0 in [('trace.csv',traces),('calibration_bins.csv',bins),('error_groups.csv',groups)]:
            csvwrite(out/fn,rr0)
    write_json(out/'completion.json',dict(complete=True,metric_rows=len(rows),seeds=seeds,stage=args.stage))
    print('Completed:',out,flush=True)


if __name__=='__main__':
    main()
