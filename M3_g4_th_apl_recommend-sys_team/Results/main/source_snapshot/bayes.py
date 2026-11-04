"""Hybrid EP/VMP for the static, identity-only Matchbox factor graph.

Gaussian moments travel forward through products. VMP messages travel back
through products; Gaussian sum-product and ordinal moment matching handle
the remaining factors. ADF discards sites; EP subtracts each old site before
replacing it. This is a transparent restricted implementation, not Infer.NET.
See EXPLANATION.md for equations, approximations and scope differences.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import ndtr


@njit(cache=True)
def truncated_moments(mean, variance, low, high):
    """Moments of N(mean, variance) restricted to (low, high].

    Compute same-sign intervals in the appropriate tail to avoid subtracting
    two CDF values near one. An underflow returns ok=False, never a made-up
    posterior. The caller records and skips this rating update.
    """
    sd = math.sqrt(variance)
    a, b = (low - mean) / sd, (high - mean) / sd
    root2 = math.sqrt(2.0)
    if a >= 0:
        z = 0.5 * (math.erfc(a / root2) - math.erfc(b / root2))
    else:
        z = 0.5 * (math.erfc(-b / root2) - math.erfc(-a / root2))
    if z < 1e-250:
        return mean, variance, z, False
    pa = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    pb = math.exp(-0.5 * b * b) / math.sqrt(2.0 * math.pi)
    apa = a * pa if math.isfinite(a) else 0.0
    bpb = b * pb if math.isfinite(b) else 0.0
    ratio = (pa - pb) / z
    factor = 1.0 + (apa - bpb) / z - ratio * ratio
    if factor <= 0.0 or not math.isfinite(factor):
        return mean, variance, z, False
    return mean + sd * ratio, variance * min(factor, 1.0), z, True


@njit(cache=True)
def local_messages(cavity_p, cavity_h, rating, offset, noise_var,
                   thresholds, inner_max, inner_tol, inner_damping):
    """Return one rating's messages to [s_1..s_K,t_1..t_K,b_u,b_i].

    Input/output use Gaussian natural parameters precision p and h=p*mean.
    Inner damping blends messages, not likelihoods. Every local marginal is
    reconstructed from the SAME cavity plus the current message, so inner
    iterations do not repeatedly multiply the observed rating into the prior.
    """
    width = len(cavity_p)
    kdim = (width - 2) // 2
    msg_p, msg_h = np.zeros(width), np.zeros(width)
    cm, cv = cavity_h / cavity_p, 1.0 / cavity_p
    m, v = cm.copy(), cv.copy()
    used, converged = 0, False
    for inner in range(inner_max):
        used = inner + 1
        zmean, zvar = np.empty(kdim), np.empty(kdim)
        sum_m, sum_v = 0.0, 0.0
        for k in range(kdim):
            j = k + kdim
            zmean[k] = m[k] * m[j]
            zvar[k] = v[k] * v[j] + v[k] * m[j] ** 2 + v[j] * m[k] ** 2
            sum_m += zmean[k]
            sum_v += zvar[k]
        # Bias-to-sum messages are cavities; only product outputs use local
        # marginals in the hybrid VMP update (Stern et al., equation 6).
        pm = offset + sum_m + cm[-2] + cm[-1]
        pv = noise_var + sum_v + cv[-2] + cv[-1]
        tm, tv, _, ok = truncated_moments(pm, pv,
                                         thresholds[rating - 1], thresholds[rating])
        if not ok:
            return msg_p, msg_h, used, False, False
        out_p = 1.0 / tv - 1.0 / pv
        out_h = tm / tv - pm / pv
        new_p, new_h = np.zeros(width), np.zeros(width)
        if out_p > 1e-14:
            obs_var, obs_mean = 1.0 / out_p, out_h / out_p
            for k in range(kdim):
                j = k + kdim
                target_var = obs_var + noise_var + sum_v - zvar[k] + cv[-2] + cv[-1]
                target_mean = obs_mean - offset - sum_m + zmean[k] - cm[-2] - cm[-1]
                # The second moment is essential: replacing it by m^2
                # silently changes VMP into a point-estimate update.
                new_p[k] = (v[j] + m[j] ** 2) / target_var
                new_h[k] = m[j] * target_mean / target_var
                new_p[j] = (v[k] + m[k] ** 2) / target_var
                new_h[j] = m[k] * target_mean / target_var
            for q in range(2):
                idx, other = width - 2 + q, width - 1 - q
                target_var = obs_var + noise_var + sum_v + cv[other]
                target_mean = obs_mean - offset - sum_m - cm[other]
                new_p[idx] = 1.0 / target_var
                new_h[idx] = target_mean / target_var
        msg_p = (1.0 - inner_damping) * msg_p + inner_damping * new_p
        msg_h = (1.0 - inner_damping) * msg_h + inner_damping * new_h
        nm, nv = (cavity_h + msg_h) / (cavity_p + msg_p), 1.0 / (cavity_p + msg_p)
        delta = max(np.max(np.abs(nm - m)), np.max(np.abs(nv - v)))
        m, v = nm, nv
        if delta < inner_tol:
            converged = True
            break
    return msg_p, msg_h, used, converged, True


@njit(cache=True)
def sweep(users, items, ratings, order, up, uh, ip, ih, sites_p, sites_h,
          keep_sites, offset, noise_var, thresholds, inner_max, inner_tol,
          inner_damping, outer_damping):
    kdim = up.shape[1] - 1
    width = 2 * kdim + 2
    skipped, capped, total_inner = 0, 0, 0
    max_delta = 0.0
    for row in order:
        u, i = users[row], items[row]
        p, h = np.empty(width), np.empty(width)
        for k in range(kdim):
            p[k], h[k] = up[u, k], uh[u, k]
            p[k + kdim], h[k + kdim] = ip[i, k], ih[i, k]
        p[-2], h[-2] = up[u, kdim], uh[u, kdim]
        p[-1], h[-1] = ip[i, kdim], ih[i, kdim]
        old_p, old_h = np.zeros(width), np.zeros(width)
        if keep_sites:
            old_p, old_h = sites_p[row].copy(), sites_h[row].copy()
        cp, ch = p - old_p, h - old_h  # EP cavity division in natural form.
        if np.min(cp) <= 0 or not np.all(np.isfinite(cp)):
            skipped += 1
            continue
        mp, mh, inner, conv, ok = local_messages(
            cp, ch, ratings[row], offset, noise_var, thresholds,
            inner_max, inner_tol, inner_damping)
        total_inner += inner
        if not ok:
            skipped += 1
            continue
        if not conv:
            capped += 1
        mp = (1.0 - outer_damping) * old_p + outer_damping * mp
        mh = (1.0 - outer_damping) * old_h + outer_damping * mh
        np_, nh = cp + mp, ch + mh
        if not np.all(np.isfinite(nh)) or not np.all(np.isfinite(np_)):
            skipped += 1
            continue
        max_delta = max(max_delta, np.max(np.abs(nh / np_ - h / p)))
        for k in range(kdim):
            up[u, k], uh[u, k] = np_[k], nh[k]
            ip[i, k], ih[i, k] = np_[k + kdim], nh[k + kdim]
        up[u, kdim], uh[u, kdim] = np_[-2], nh[-2]
        ip[i, kdim], ih[i, kdim] = np_[-1], nh[-1]
        if keep_sites:
            sites_p[row], sites_h[row] = mp, mh
    return skipped, capped, total_inner, max_delta


def gaussian_rating_probabilities(mean, variance, thresholds=None):
    if thresholds is None:
        thresholds = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])
    z = (thresholds[None, :] - np.asarray(mean)[:, None]) / np.sqrt(np.asarray(variance))[:, None]
    lo, hi = z[:, :-1], z[:, 1:]
    p = np.where(lo >= 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    p = np.maximum(p, 1e-15)
    return p / p.sum(axis=1, keepdims=True)

# Procedural state: p = precision = 1/variance, h = mean/variance.
# The three numerical kernels above are unchanged from the recovered M2 code.
def init(nu, ni, k, seed, r, mode, noise=1.0, prior_var=0.25):
    rng = np.random.default_rng(seed)
    up, ip = np.full((nu, k + 1), 1.0/prior_var), np.full((ni, k + 1), 1.0/prior_var)
    up[:, -1], ip[:, -1] = 1.0, 1.0
    uh = rng.normal(0, 0.1, up.shape) * up
    ih = rng.normal(0, 0.1, ip.shape) * ip
    uh[:, -1], ih[:, -1] = 0.0, 0.0
    n = len(r) if mode == 'ep' else 0
    return dict(up=up, uh=uh, ip=ip, ih=ih, k=k, mode=mode, passes=0,
                sp=np.zeros((n, 2 * k + 2)), sh=np.zeros((n, 2 * k + 2)),
                mu=float(np.mean(r)), noise=float(noise),
                th=np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf]))


def step(m, u, i, r, order, outer_damping=0.5, inner_damping=0.7):
    if m['mode'] == 'adf' and m['passes']:
        raise ValueError('ADF gets one pass. Replaying ADF is not EP.')
    d = sweep(u, i, r, order, m['up'], m['uh'], m['ip'], m['ih'],
              m['sp'], m['sh'], m['mode'] == 'ep', m['mu'], m['noise'],
              m['th'], 10, 1e-4, inner_damping, 1.0 if m['passes'] == 0 else outer_damping)
    m['passes'] += 1
    return dict(skipped_updates=int(d[0]), capped_updates=int(d[1]),
                mean_inner_iterations=float(d[2] / len(r)),
                max_mean_change=float(d[3]),
                site_mib=(m['sp'].nbytes + m['sh'].nbytes) / 2**20)


def prob(m, u, i):
    um, uv = m['uh'][u] / m['up'][u], 1.0 / m['up'][u]
    im, iv = m['ih'][i] / m['ip'][i], 1.0 / m['ip'][i]
    mu = m['mu'] + um[:, -1] + im[:, -1] + np.sum(um[:, :-1] * im[:, :-1], axis=1)
    var = uv[:, -1] + iv[:, -1] + np.sum(
        uv[:, :-1] * iv[:, :-1] + uv[:, :-1] * im[:, :-1]**2
        + iv[:, :-1] * um[:, :-1]**2, axis=1)
    return gaussian_rating_probabilities(mu, var + m['noise'], m['th'])


def save(m, path):
    # Includes sites for inspection; training arrays/order live in split files.
    np.savez_compressed(path, **m)
