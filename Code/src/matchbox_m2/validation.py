"""Independent mathematical checks plus algorithmic invariants (no dataset)."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from .data import split_indices, write_json
from .metrics import evaluate
from .model import Matchbox, gaussian_rating_probabilities, local_messages, truncated_moments


class MathematicalTests(unittest.TestCase):
    def test_truncation_against_independent_quadrature(self):
        for mean, var, low, high in [(0.4, 1.7, -np.inf, 1.5), (3.2, 0.7, 2.5, 3.5),
                                      (0, 1, 4, 5), (4, 2, 4.5, np.inf)]:
            f = lambda x: norm.pdf(x, loc=mean, scale=np.sqrt(var))
            z = quad(f, low, high, epsabs=1e-13)[0]
            qm = quad(lambda x: x * f(x), low, high, epsabs=1e-13)[0] / z
            qv = quad(lambda x: (x - qm) ** 2 * f(x), low, high, epsabs=1e-13)[0] / z
            m, v, p, ok = truncated_moments(mean, var, low, high)
            self.assertTrue(ok)
            np.testing.assert_allclose([m, v, p], [qm, qv, z], atol=2e-10, rtol=2e-9)

    def test_k0_single_factor_matches_exact_marginals(self):
        # With K=0 the two biases and noisy score are jointly Gaussian.
        # Gaussian conditioning on an interval gives exact individual moments.
        mean, var = np.array([0.2, -0.3]), np.array([1.1, 0.7])
        offset, beta2, rating = 3.0, 0.8, 4
        thresholds = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])
        total_mean, total_var = offset + sum(mean), sum(var) + beta2
        tm, tv, _, _ = truncated_moments(total_mean, total_var, 3.5, 4.5)
        exact_m = mean + var / total_var * (tm - total_mean)
        exact_v = var + var ** 2 / total_var ** 2 * (tv - total_var)
        cp, ch = 1 / var, mean / var
        mp, mh, _, _, ok = local_messages(cp, ch, rating, offset, beta2, thresholds, 20, 1e-12, 1.0)
        self.assertTrue(ok)
        np.testing.assert_allclose((ch + mh) / (cp + mp), exact_m, atol=1e-12)
        np.testing.assert_allclose(1 / (cp + mp), exact_v, atol=1e-12)

    def test_first_pass_ep_equals_adf(self):
        u = np.array([0, 0, 1, 2, 1, 2])
        i = np.array([0, 1, 0, 1, 2, 2])
        r = np.array([1, 4, 3, 5, 2, 4])
        models = [Matchbox(3, 3, k=2, seed=7) for _ in range(2)]
        for model, mode in zip(models, ["adf", "ep"]):
            model.initialize_training(r, mode)
            model.fit_pass(u, i, r, np.arange(len(r)))
        for name in ["up", "uh", "ip", "ih"]:
            np.testing.assert_array_equal(getattr(models[0], name), getattr(models[1], name))
        with self.assertRaises(ValueError):
            models[0].fit_pass(u, i, r, np.arange(len(r)))

    def test_prior_plus_sites_equals_global_natural_parameters(self):
        u, i, r = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.array([1, 5, 4, 2])
        model = Matchbox(2, 2, k=2)
        priors = [x.copy() for x in [model.up, model.uh, model.ip, model.ih]]
        model.initialize_training(r, "ep")
        for _ in range(4):
            model.fit_pass(u, i, r, np.arange(4))
        up, uh, ip, ih = priors
        for row, (a, b) in enumerate(zip(u, i)):
            up[a, :2] += model.sites_p[row, :2]; uh[a, :2] += model.sites_h[row, :2]
            ip[b, :2] += model.sites_p[row, 2:4]; ih[b, :2] += model.sites_h[row, 2:4]
            up[a, -1] += model.sites_p[row, -2]; uh[a, -1] += model.sites_h[row, -2]
            ip[b, -1] += model.sites_p[row, -1]; ih[b, -1] += model.sites_h[row, -1]
        for actual, expected in zip([model.up, model.uh, model.ip, model.ih], [up, uh, ip, ih]):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_one_rating_revisit_does_not_double_count(self):
        model = Matchbox(1, 1, k=2, outer_damping=1.0)
        u, i, r = np.array([0]), np.array([0]), np.array([4])
        model.initialize_training(r)
        model.fit_pass(u, i, r, np.array([0]))
        original = [x.copy() for x in [model.up, model.uh, model.ip, model.ih]]
        for _ in range(5):
            model.fit_pass(u, i, r, np.array([0]))
        for a, b in zip(original, [model.up, model.uh, model.ip, model.ih]):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_product_prediction_moments_against_monte_carlo(self):
        model = Matchbox(1, 1, k=3, seed=11)
        rng = np.random.default_rng(13)
        n = 300000
        u = rng.normal(model.uh[0] / model.up[0], np.sqrt(1 / model.up[0]), (n, 4))
        i = rng.normal(model.ih[0] / model.ip[0], np.sqrt(1 / model.ip[0]), (n, 4))
        x = u[:, -1] + i[:, -1] + (u[:, :3] * i[:, :3]).sum(axis=1)
        mean, _, epistemic = model.latent_moments(np.array([0]), np.array([0]))
        self.assertLess(abs(x.mean() - mean[0]), 0.015)
        self.assertLess(abs(x.var() - epistemic[0]), 0.03)

    def test_probabilities_and_metrics(self):
        p = gaussian_rating_probabilities(np.array([-20., 3., 20.]), np.ones(3))
        np.testing.assert_allclose(p.sum(axis=1), 1)
        self.assertTrue(np.all(np.isfinite(p)) and np.all(p > 0))
        r = np.array([1, 3, 5])
        exact = np.eye(5)[r - 1]
        values = evaluate(r, exact)
        for key in ["mae", "rmse", "nll", "brier", "ece"]:
            self.assertAlmostEqual(values[key], 0)

    def test_splits_disjoint_complete_reproducible(self):
        a = split_indices(101, 42)
        b = split_indices(101, 42)
        np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(101))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[0], split_indices(101, 43)[0]))

    def test_checkpoint_roundtrip(self):
        model = Matchbox(2, 3, k=2)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "model.npz"
            model.save(p)
            loaded = Matchbox.load(p)
            u, i = np.array([0, 1]), np.array([1, 2])
            np.testing.assert_array_equal(model.predict_proba(u, i), loaded.predict_proba(u, i))


def run_checks(path=None):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MathematicalTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = dict(tests_run=result.testsRun, failures=len(result.failures),
                    errors=len(result.errors), successful=result.wasSuccessful(),
                    tests=unittest.defaultTestLoader.getTestCaseNames(MathematicalTests),
                    scope="Exact local Gaussian/ordinal cases and implementation invariants; not a proof of global EP convergence or exact bilinear inference")
    if path:
        write_json(path, evidence)
    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run_checks(args.output) else 1)
