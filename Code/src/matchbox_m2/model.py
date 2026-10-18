"""Hybrid EP/VMP for the static, identity-only Matchbox factor graph.

Gaussian moments travel forward through products. VMP messages travel back
through products; Gaussian sum-product and ordinal moment matching handle
the remaining factors. ADF discards sites; EP subtracts each old site before
replacing it. This is a transparent restricted implementation, not Infer.NET.
See Report/latex/main.tex for equations, approximations and scope differences.
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


class Matchbox:
    def __init__(self, n_users, n_items, k=8, seed=42, noise_var=1.0,
                 trait_var=0.25, bias_var=1.0, init_scale=0.1,
                 inner_max=10, inner_tol=1e-4, inner_damping=0.7,
                 outer_damping=0.5):
        self.k, self.seed = k, seed
        self.noise_var = float(noise_var)
        self.inner_max, self.inner_tol = inner_max, inner_tol
        self.inner_damping, self.outer_damping = inner_damping, outer_damping
        self.thresholds = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])
        rng = np.random.default_rng(seed)
        self.up = np.full((n_users, k + 1), 1.0 / trait_var)
        self.ip = np.full((n_items, k + 1), 1.0 / trait_var)
        self.up[:, -1], self.ip[:, -1] = 1.0 / bias_var, 1.0 / bias_var
        # Small, seeded PRIOR means break the bilinear sign symmetry. They
        # are a disclosed modelling assumption, independent of held-out labels.
        self.uh = rng.normal(0, init_scale, self.up.shape) * self.up
        self.ih = rng.normal(0, init_scale, self.ip.shape) * self.ip
        self.uh[:, -1], self.ih[:, -1] = 0.0, 0.0
        self.offset = 0.0
        self.passes = 0
        self.sites_p = np.empty((0, 0))
        self.sites_h = np.empty((0, 0))

    def initialize_training(self, ratings, mode="ep"):
        if mode not in ("adf", "ep"):
            raise ValueError("mode must be adf or ep")
        self.mode = mode
        self.offset = float(np.mean(ratings))  # train-only plug-in global mean
        n = len(ratings) if mode == "ep" else 0
        self.sites_p = np.zeros((n, 2 * self.k + 2))
        self.sites_h = np.zeros_like(self.sites_p)

    def fit_pass(self, users, items, ratings, order):
        if self.mode == "adf" and self.passes:
            raise ValueError("ADF is single-pass; replay is not EP")
        diagnostics = sweep(users, items, ratings, order,
                            self.up, self.uh, self.ip, self.ih,
                            self.sites_p, self.sites_h, self.mode == "ep",
                            self.offset, self.noise_var, self.thresholds,
                            self.inner_max, self.inner_tol, self.inner_damping,
                            1.0 if self.passes == 0 else self.outer_damping)
        self.passes += 1
        return dict(skipped_updates=int(diagnostics[0]), capped_updates=int(diagnostics[1]),
                    mean_inner_iterations=float(diagnostics[2] / len(ratings)),
                    max_parameter_mean_change=float(diagnostics[3]),
                    site_memory_bytes=int(self.sites_p.nbytes + self.sites_h.nbytes))

    def latent_moments(self, users, items):
        um, uv = self.uh[users] / self.up[users], 1.0 / self.up[users]
        im, iv = self.ih[items] / self.ip[items], 1.0 / self.ip[items]
        mean = self.offset + um[:, -1] + im[:, -1] + np.sum(um[:, :-1] * im[:, :-1], axis=1)
        epistemic = uv[:, -1] + iv[:, -1] + np.sum(
            uv[:, :-1] * iv[:, :-1] + uv[:, :-1] * im[:, :-1] ** 2
            + iv[:, :-1] * um[:, :-1] ** 2, axis=1)
        return mean, epistemic + self.noise_var, epistemic

    def predict_proba(self, users, items):
        mean, variance, _ = self.latent_moments(users, items)
        return gaussian_rating_probabilities(mean, variance, self.thresholds)

    def save(self, path):
        np.savez_compressed(path, up=self.up, uh=self.uh, ip=self.ip, ih=self.ih,
                            offset=self.offset, noise_var=self.noise_var,
                            thresholds=self.thresholds, k=self.k, passes=self.passes)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as d:
            model = cls(d["up"].shape[0], d["ip"].shape[0], int(d["k"]))
            for name in ["up", "uh", "ip", "ih", "thresholds"]:
                setattr(model, name, d[name].copy())
            model.offset, model.noise_var, model.passes = float(d["offset"]), float(d["noise_var"]), int(d["passes"])
        return model


def gaussian_rating_probabilities(mean, variance, thresholds=None):
    if thresholds is None:
        thresholds = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])
    z = (thresholds[None, :] - np.asarray(mean)[:, None]) / np.sqrt(np.asarray(variance))[:, None]
    lo, hi = z[:, :-1], z[:, 1:]
    p = np.where(lo >= 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    p = np.maximum(p, 1e-15)
    return p / p.sum(axis=1, keepdims=True)
