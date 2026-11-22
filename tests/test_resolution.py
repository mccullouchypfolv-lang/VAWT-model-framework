"""Continuous evaluation of training-resolved periodic inputs and sources."""
import copy
import importlib.util
import unittest

import numpy as np

from lf_operator.spectral_source import sample_source


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('lf_operator.resolution'),
                             'resolution evaluation module must exist')
        from lf_operator import resolution
        self.r = resolution

    def test_midpoint_grid_and_invalid_counts(self):
        np.testing.assert_array_equal(self.r.midpoint_theta(160),
                                      (np.arange(160) + .5) * (2*np.pi/160))
        for n in (True, False, 3, 0, -2, 160., '160'):
            with self.subTest(n=n), self.assertRaises(ValueError):
                self.r.midpoint_theta(n)

    def test_profile_analytic_multichannel_up_and_down_odd_grid(self):
        reference = self.r.midpoint_theta(160)
        def profile(t):
            return np.stack((1.2 + np.sin(3*t) - .3*np.cos(17*t),
                             .7*np.cos(4*t) + np.sin(31*t)), axis=1)
        values = profile(reference)
        for n in (320, 640, 79):
            theta = self.r.midpoint_theta(n)
            np.testing.assert_allclose(self.r.evaluate_periodic_profile(values, reference, theta),
                                       profile(theta), atol=3e-14)
        np.testing.assert_array_equal(self.r.evaluate_periodic_profile(values, reference, reference),
                                      values)
        np.testing.assert_allclose(self.r.evaluate_periodic_profile(values[:, 0], reference,
                                                                   self.r.midpoint_theta(640)),
                                   profile(self.r.midpoint_theta(640))[:, 0], atol=3e-14)

    def test_reference_offset_odd_modes_and_nyquist_cosine(self):
        reference = np.arange(15) * (2*np.pi/15)
        theta = self.r.midpoint_theta(47)
        values = 2 + np.cos(7*reference) - .2*np.sin(6*reference)
        np.testing.assert_allclose(self.r.evaluate_periodic_profile(values, reference, theta),
                                   2 + np.cos(7*theta) - .2*np.sin(6*theta), atol=2e-14)
        reference = self.r.midpoint_theta(16)
        values = (-1.)**np.arange(16)
        for n in (16, 32, 64, 15):
            theta = self.r.midpoint_theta(n)
            np.testing.assert_allclose(self.r.evaluate_periodic_profile(values, reference, theta),
                                       np.cos(8*(theta-reference[0])), atol=2e-14)

    def test_source_exact_legacy_native_and_same_path_on_query_grids(self):
        spec = {'n_theta': 16, 'psd': [.4, .2, .1, .3, .1, .2, .1, .1, .7],
                'nu': .8, 'kappa': 3., 'amplitude': .9}
        unchanged = copy.deepcopy(spec)
        native_theta = self.r.midpoint_theta(16)
        original = sample_source(spec, 7, rng=np.random.default_rng(4))
        native = self.r.evaluate_source(spec, 7, native_theta, np.random.default_rng(4))
        np.testing.assert_array_equal(native, original)
        rounded_theta = native_theta.astype(np.float32)
        np.testing.assert_array_equal(self.r.evaluate_source(
            spec, 7, rounded_theta, np.random.default_rng(4)), original)
        np.testing.assert_array_equal(self.r.evaluate_periodic_profile(
            original[0], rounded_theta, native_theta), original[0])
        for n in (32, 64, 15):
            theta = self.r.midpoint_theta(n)
            actual = self.r.evaluate_source(spec, 7, theta, np.random.default_rng(4))
            expected = self.r.evaluate_periodic_profile(original[..., 0].T,
                                                        native_theta, theta).T[..., None]
            np.testing.assert_allclose(actual, expected, atol=2e-14)
        # Odd refinement shares original midpoint locations exactly.
        fine = self.r.evaluate_source(spec, 7, self.r.midpoint_theta(48), np.random.default_rng(4))
        np.testing.assert_allclose(fine[:, 1::3], native, atol=2e-14)
        training_spec = {'n_theta': 160, 'psd': (1 + np.arange(81)**2.)**-1.3}
        coarse = self.r.evaluate_source(training_spec, 8, self.r.midpoint_theta(160),
                                       np.random.default_rng(42))
        refined = self.r.evaluate_source(training_spec, 8, self.r.midpoint_theta(480),
                                        np.random.default_rng(42))
        np.testing.assert_allclose(refined[:, 1::3], coarse, atol=4e-14)
        self.assertEqual(spec, unchanged)
        self.assertEqual(self.r.SOURCE_SCOPE, 'training-resolved periodic Matérn process')

    def test_source_resolved_covariance_and_no_frequency_renormalization(self):
        n = 16
        psd = np.array([.4, .2, .1, .3, .1, .2, .1, .1, .7])
        spec = {'n_theta': n, 'psd': psd.tolist()}
        theta = self.r.midpoint_theta(32)
        draws = self.r.evaluate_source(spec, 40000, theta, np.random.default_rng(7))[..., 0]
        delta = theta[:, None] - theta[None, :]
        expected = np.full((32, 32), psd[0])
        for k in range(1, n//2):
            expected += 2*psd[k]*np.cos(k*delta)
        nyquist = np.cos((n//2)*(theta-np.pi/n))
        expected += psd[-1]*np.outer(nyquist, nyquist)
        np.testing.assert_allclose(draws.T @ draws / len(draws), expected, atol=.09)
        modes = np.fft.rfft(draws, axis=1, norm='forward')
        np.testing.assert_allclose(np.mean(abs(modes[:, :8])**2, 0), psd[:8], rtol=.035)
        self.assertAlmostEqual(np.mean(abs(modes[:, 8])**2), psd[8]/4, delta=.01)
        self.assertLess(np.max(abs(modes[:, 9:])), 1e-13)

    def test_invalid_profiles_grids_and_sources(self):
        theta = self.r.midpoint_theta(16)
        for invalid in (theta[:-1], theta[::-1], theta + 2*np.pi,
                        np.linspace(0, 2*np.pi, 16), np.zeros(16),
                        np.array([0., 1., 2., np.nan]), theta.astype(complex)):
            with self.subTest(grid=invalid), self.assertRaises(ValueError):
                self.r.evaluate_periodic_profile(np.ones(16), theta, invalid)
        for values in (np.ones(15), np.ones((16, 1, 1)), np.ones(16)*np.nan,
                       np.ones(16, dtype=complex)):
            with self.assertRaises(ValueError):
                self.r.evaluate_periodic_profile(values, theta, theta)
        with self.assertRaises(ValueError):
            self.r.evaluate_source({'n_theta': 16, 'psd': [1.]*9}, 2, theta, 4)
        with self.assertRaises(ValueError):
            self.r.evaluate_source({'n_theta': 16, 'psd': [1.]*9}, 0, theta,
                                   np.random.default_rng(1))
        with self.assertRaises(ValueError):
            self.r.evaluate_source({'n_theta': 16, 'psd': [1.]*9, 'domain': [0., 1.]},
                                   2, theta, np.random.default_rng(1))


if __name__ == '__main__':
    unittest.main()
