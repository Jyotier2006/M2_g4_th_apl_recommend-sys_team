"""A numerical ground-truth reference for one bilinear, ordinal observation.

Gauss-Hermite quadrature is independent of the EP/VMP update. It is feasible
here because there are only two non-negligible continuous latent variables.
It is not a scalable exact inference method for the MovieLens graph.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr

from .model import local_messages


def reference(n):
    nodes, weights = hermgauss(n)
    ms, mt, vs, vt = 0.5, 0.8, 0.4, 0.4
    s = ms + np.sqrt(2 * vs) * nodes[:, None]
    t = mt + np.sqrt(2 * vt) * nodes[None, :]
    w = weights[:, None] * weights[None, :] / np.pi
    # Rating five means latent y > 4.5; independent tiny biases integrate out.
    likelihood = ndtr((3.0 + s * t - 4.5) / np.sqrt(1.0 + 2e-8))
    q = w * likelihood
    z = q.sum(); q /= z
    sm, tm = float((q * s).sum()), float((q * t).sum())
    return np.array([sm, float((q * (s-sm)**2).sum()), tm, float((q * (t-tm)**2).sum()),
                     float((q * (s-sm) * (t-tm)).sum())])


def run_study(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    low, high = reference(40), reference(80)
    cp = 1 / np.array([0.4, 0.4, 1e-8, 1e-8])
    ch = cp * np.array([0.5, 0.8, 0., 0.])
    thresholds = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])
    mp, mh, used, converged, ok = local_messages(cp, ch, 5, 3.0, 1.0, thresholds, 500, 1e-12, 0.5)
    if not ok or not converged or np.max(np.abs(high-low)) > 1e-6:
        raise AssertionError("Numerical reference or local solver did not stabilize")
    m, v = (ch + mh) / (cp + mp), 1 / (cp + mp)
    approx = np.array([m[0], v[0], m[1], v[1], 0.0])
    frame = pd.DataFrame(dict(quantity=["user_mean", "user_variance", "item_mean", "item_variance", "user_item_covariance"],
                              quadrature40=low, quadrature80=high, hybrid=approx,
                              absolute_error=np.abs(high-approx), quadrature_change=np.abs(high-low)))
    frame.to_csv(output / "bilinear_validation.csv", index=False)
    return dict(inner_iterations=int(used), converged=bool(converged), max_quadrature_change=float(np.max(np.abs(high-low))))
