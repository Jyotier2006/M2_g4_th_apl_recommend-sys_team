"""MAP matrix factorization with a proper Gaussian Markov movie prior.

No EP/ADF here: alternating exact block minimization (Gauss-Seidel).
The graph is built using TRAIN ratings and optional fixed movie genres.
"""
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy import sparse
from scipy.optimize import minimize_scalar

from bayes import gaussian_rating_probabilities


def movie_info(root, iid):
    rows = {}
    for line in (Path(root) / 'movies.dat').read_text(encoding='latin-1').splitlines():
        mid, title, genres = line.split('::')
        rows[int(mid)] = (title, genres.split('|'))
    gs = sorted({g for _, genres in rows.values() for g in genres})
    mp = {g: j for j, g in enumerate(gs)}
    x, titles = np.zeros((len(iid), len(gs))), []
    for j, mid in enumerate(iid):
        title, genres = rows[int(mid)]
        titles.append(title)
        for g in genres:
            x[j, mp[g]] = 1
    return x, titles, gs


def graph(u, i, r, nu, ni, genres, top=20, mix=0.1):
    """Symmetric nonnegative top-neighbour graph; no validation/test input.

    1. Subtract a train-only regularized user/movie bias baseline.
    2. Cosine similarity of residual movie columns, with common-user shrinkage.
    3. Optional genre cosine mixture, choose top neighbours, symmetrize.
    4. Symmetric degree normalization controls the graph-penalty scale.
    """
    mu = r.mean()
    bu, bi = np.zeros(nu), np.zeros(ni)
    cu, ci = np.bincount(u, minlength=nu), np.bincount(i, minlength=ni)
    for _ in range(5):
        bu = np.bincount(u, weights=r-mu-bi[i], minlength=nu) / (cu + 10)
        bi = np.bincount(i, weights=r-mu-bu[u], minlength=ni) / (ci + 10)
    e = r - mu - bu[u] - bi[i]
    a = sparse.csr_matrix((e, (u, i)), shape=(nu, ni))
    norm = np.sqrt(np.asarray(a.power(2).sum(axis=0)).ravel())
    a = a @ sparse.diags(1 / np.maximum(norm, 1e-12))
    seen = sparse.csr_matrix((np.ones(len(r)), (u, i)), shape=(nu, ni))
    gx = genres / np.maximum(np.linalg.norm(genres, axis=1, keepdims=True), 1e-12)
    rs, cs, vs = [], [], []
    # Batch rows: avoids keeping dense ni x ni similarity/count matrices.
    for lo in range(0, ni, 128):
        hi = min(lo + 128, ni)
        co = (seen[:, lo:hi].T @ seen).toarray()
        sim = (a[:, lo:hi].T @ a).toarray()
        sim = np.maximum(sim, 0) * co / (co + 20)
        sim[co < 5] = 0
        sim = (1-mix) * sim + mix * (gx[lo:hi] @ gx.T)
        for q, row in enumerate(sim):
            j = lo + q
            row[j] = 0
            # Stable tie break by mapped movie ID; at most top outgoing edges.
            js = np.lexsort((np.arange(ni), -row))[:min(top, ni-1)]
            js = js[row[js] > 0]
            rs.extend([j] * len(js)); cs.extend(js); vs.extend(row[js])
    w = sparse.csr_matrix((vs, (rs, cs)), shape=(ni, ni))
    w = w.maximum(w.T)
    degree = np.asarray(w.sum(axis=1)).ravel()
    d = sparse.diags(1 / np.sqrt(np.maximum(degree, 1e-12)))
    w = (d @ w @ d).tocsr()
    w.sort_indices()
    return w


def init(nu, ni, k, seed, r):
    rng = np.random.default_rng(seed)
    p, q = rng.normal(0, 0.1, (nu, k)), rng.normal(0, 0.1, (ni, k))
    return dict(p=p, q=q, bu=np.zeros(nu), bi=np.zeros(ni), mu=float(r.mean()))


def grouped(u, i, r, n):
    order = np.argsort(u, kind='stable')
    ptr = np.r_[0, np.cumsum(np.bincount(u, minlength=n))]
    return ptr.astype(np.int64), i[order], r[order].astype(np.float64)


@njit(cache=True)
def als(uptr, ui, ur, iptr, iu, ir, wp, wi, wv, degree,
        p, q, bu, bi, mu, reg, breg, gamma):
    k = p.shape[1]
    # One exact user block update; bias is included as the last coordinate.
    for u in range(len(bu)):
        aa = np.eye(k+1) * reg
        aa[k, k] = breg
        bb = np.zeros(k+1)
        for z in range(uptr[u], uptr[u+1]):
            j = ui[z]
            x = np.empty(k+1)
            x[:k], x[k] = q[j], 1.0
            y = ur[z] - mu - bi[j]
            aa += np.outer(x, x)
            bb += x * y
        sol = np.linalg.solve(aa, bb)
        p[u], bu[u] = sol[:k], sol[k]
    # Sequential movie updates use the newest available neighbours (Gauss-Seidel).
    for j in range(len(bi)):
        aa = np.eye(k+1) * reg
        for t in range(k):
            aa[t, t] += gamma * degree[j]
        aa[k, k] = breg
        bb = np.zeros(k+1)
        for z in range(iptr[j], iptr[j+1]):
            u = iu[z]
            x = np.empty(k+1)
            x[:k], x[k] = p[u], 1.0
            y = ir[z] - mu - bu[u]
            aa += np.outer(x, x)
            bb += x * y
        for z in range(wp[j], wp[j+1]):
            for t in range(k):
                bb[t] += gamma * wv[z] * q[wi[z], t]
        sol = np.linalg.solve(aa, bb)
        q[j], bi[j] = sol[:k], sol[k]


def point(m, u, i):
    return m['mu'] + m['bu'][u] + m['bi'][i] + np.sum(m['p'][u] * m['q'][i], axis=1)


def prob(m, u, i):
    # Validation-fitted observation scale; NOT posterior parameter uncertainty.
    return gaussian_rating_probabilities(point(m, u, i), np.full(len(u), m['noise']))


def calibrate(m, u, i, r):
    mean = point(m, u, i)
    def f(x):
        pr = gaussian_rating_probabilities(mean, np.full(len(u), np.exp(2*x)))
        return float(-np.log(pr[np.arange(len(r)), r-1]).mean())
    res = minimize_scalar(f, bounds=(np.log(0.2), np.log(3)), method='bounded')
    m['noise'] = float(np.exp(2*res.x))


def objective(m, u, i, r, w, reg, breg, gamma):
    q = m['q']
    deg = np.asarray(w.sum(axis=1)).ravel()
    edge = np.sum(deg[:, None] * q*q) - np.sum(q * (w @ q))
    return float(0.5*np.sum((r-point(m,u,i))**2)
                 + 0.5*reg*(np.sum(m['p']**2)+np.sum(q**2))
                 + 0.5*breg*(np.sum(m['bu']**2)+np.sum(m['bi']**2))
                 + 0.5*gamma*edge)


def fit(u, i, r, vu, vi, vr, nu, ni, k, seed, w, gamma,
        epochs=25, reg=10.0, breg=10.0):
    gamma,reg,breg=float(gamma),float(reg),float(breg)
    m = init(nu, ni, k, seed, r)
    ud = grouped(u, i, r, nu)
    it = grouped(i, u, r, ni)
    degree = np.asarray(w.sum(axis=1)).ravel()
    wp,wi=w.indptr.astype(np.int64),w.indices.astype(np.int64)
    best, state, trace = np.inf, None, []
    start = time.perf_counter()
    for ep in range(1, epochs+1):
        als(*ud, *it, wp, wi, w.data, degree,
            m['p'], m['q'], m['bu'], m['bi'], m['mu'], reg, breg, gamma)
        pred = np.clip(point(m, vu, vi), 1, 5)
        rmse = float(np.sqrt(np.mean((pred-vr)**2)))
        trace.append(dict(epoch=ep, validation_rmse=rmse,
                          objective=objective(m,u,i,r,w,reg,breg,gamma)))
        if rmse < best:
            best = rmse
            state = {key: x.copy() if isinstance(x,np.ndarray) else x for key,x in m.items()}
            state['epoch'] = ep
    state.update(validation_rmse=best, gamma=gamma)
    calibrate(state, vu, vi, vr)
    return state, trace, time.perf_counter()-start
