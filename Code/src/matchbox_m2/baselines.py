"""M1 baselines: mean predictors and conventional biased matrix factorization."""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar

from .model import gaussian_rating_probabilities


class MeanBaseline:
    """Smoothed empirical histograms; expectations are shrinkage mean ratings."""
    def __init__(self, kind="global", strength=10.0):
        self.kind, self.strength = kind, strength

    def fit(self, u, i, r, n_users, n_items):
        hist = np.bincount(r, minlength=6)[1:].astype(float)
        self.global_p = hist / hist.sum()
        if self.kind != "global":
            ids, count = (u, n_users) if self.kind == "user" else (i, n_items)
            table = np.tile(self.strength * self.global_p, (count, 1))
            np.add.at(table, (ids, r - 1), 1)
            self.table = table / table.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, u, i):
        if self.kind == "global":
            return np.tile(self.global_p, (len(u), 1))
        return self.table[u if self.kind == "user" else i]


@njit(cache=True)
def sgd_epoch(u, i, r, order, um, im, ub, ib, offset, lr, reg):
    for row in order:
        a, b = u[row], i[row]
        error = r[row] - offset - ub[a] - ib[b] - np.dot(um[a], im[b])
        ub[a] += lr * (error - reg * ub[a])
        ib[b] += lr * (error - reg * ib[b])
        for k in range(um.shape[1]):
            old_u, old_i = um[a, k], im[b, k]
            um[a, k] += lr * (error * old_i - reg * old_u)
            im[b, k] += lr * (error * old_u - reg * old_i)


class MatrixFactorization:
    def __init__(self, n_users, n_items, k=8, seed=42, lr=0.007, reg=0.03):
        rng = np.random.default_rng(seed)
        self.um, self.im = rng.normal(0, 0.1, (n_users, k)), rng.normal(0, 0.1, (n_items, k))
        self.ub, self.ib = np.zeros(n_users), np.zeros(n_items)
        self.offset, self.lr, self.reg, self.noise_var = 0.0, lr, reg, 1.0

    def point_predict(self, u, i):
        return self.offset + self.ub[u] + self.ib[i] + np.sum(self.um[u] * self.im[i], axis=1)

    def fit_epoch(self, u, i, r, order, epoch):
        self.offset = float(np.mean(r))
        sgd_epoch(u, i, r, order, self.um, self.im, self.ub, self.ib,
                  self.offset, self.lr * 0.98 ** epoch, self.reg)

    def calibrate(self, u, i, r):
        """Fit one observation scale on validation only, not a Bayesian posterior."""
        mean = self.point_predict(u, i)
        def objective(log_sd):
            p = gaussian_rating_probabilities(mean, np.full(len(mean), np.exp(2 * log_sd)))
            return float(-np.log(p[np.arange(len(r)), r - 1]).mean())
        result = minimize_scalar(objective, bounds=(np.log(0.2), np.log(3.0)), method="bounded")
        self.noise_var = float(np.exp(2 * result.x))

    def predict_proba(self, u, i):
        return gaussian_rating_probabilities(self.point_predict(u, i), np.full(len(u), self.noise_var))
