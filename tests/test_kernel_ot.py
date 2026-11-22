import json
import unittest

import numpy as np

from lf_operator.kernel_ot import (
    SinkhornConvergenceError, couple_endpoints, draw_plan_pairs,
    kernel_cost, sinkhorn_log,
)


class KernelCostTests(unittest.TestCase):
    def test_three_kernels_are_finite_symmetric_and_zero_on_equal_curves(self):
        x = np.random.default_rng(2).normal(size=(5, 12, 2))
        for name in ('rbf', 'sobolev_rbf', 'signature_truncated_rbf'):
            with self.subTest(kernel=name):
                cost, meta = kernel_cost(x, x, kernel=name)
                np.testing.assert_allclose(cost, cost.T, atol=1e-12)
                np.testing.assert_allclose(np.diag(cost), 0, atol=1e-12)
                self.assertTrue(np.isfinite(cost).all())
                self.assertAlmostEqual(np.median(cost[cost > 0]), 1.)
                self.assertEqual(meta['kernel'], name)
                json.dumps(meta, allow_nan=False)

    def test_constants_and_bounded_raw_rbf(self):
        x = np.zeros((3, 8, 1))
        for name in ('rbf', 'sobolev_rbf', 'signature_truncated_rbf'):
            cost, _ = kernel_cost(x, x, kernel=name)
            np.testing.assert_array_equal(cost, 0)
        cost, meta = kernel_cost(x, np.ones((2, 8, 1)), bandwidth=1., median_normalize=False)
        np.testing.assert_allclose(cost, 2 - 2 * np.exp(-.5))
        self.assertLessEqual(cost.max(), 2.)
        self.assertFalse(meta['median_normalize'])

    def test_sobolev_periodic_derivative_uses_radians(self):
        theta = np.arange(32) * (2 * np.pi / 32)
        x = np.sin(theta)[None, :, None]
        z = np.zeros_like(x)
        cost, _ = kernel_cost(x, z, kernel='sobolev_rbf', bandwidth=1.,
                              derivative_weight=.3, median_normalize=False)
        derivative = (np.roll(x, -1, axis=1) - np.roll(x, 1, axis=1)) / (4 * np.pi / 32)
        distance = np.mean(x*x) + .3*np.mean(derivative*derivative)
        np.testing.assert_allclose(cost, 2-2*np.exp(-distance/2))

    def test_time_augmented_signature_distinguishes_same_endpoint_shape(self):
        x = np.array([[[0.], [1.], [0.]], [[0.], [-1.], [0.]]])
        cost, meta = kernel_cost(x, x, kernel='signature_truncated_rbf', bandwidth=1.)
        self.assertGreater(cost[0, 1], 0)
        self.assertTrue(meta['signature_is_truncated_approximation'])
        cost, meta = kernel_cost(x, x, kernel='signature_truncated_rbf', time_aug=False)
        np.testing.assert_allclose(cost, 0, atol=1e-12)
        self.assertTrue(meta['signature_1d_endpoint_degeneracy'])

    def test_invalid_inputs(self):
        x = np.zeros((2, 4, 1))
        for kwargs in ({'kernel': 'bad'}, {'bandwidth': 0}, {'signature_depth': 0},
                       {'derivative_weight': -1}):
            with self.assertRaises(ValueError):
                kernel_cost(x, x, **kwargs)
        with self.assertRaises(ValueError):
            kernel_cost(x + np.nan, x)

    def test_extreme_positive_bandwidth_remains_finite(self):
        x = np.zeros((2, 4, 1))
        y = np.ones((3, 4, 1))
        for bandwidth in (1e-300, 1e300):
            cost, metadata = kernel_cost(x, y, bandwidth=bandwidth, median_normalize=False)
            self.assertTrue(np.isfinite(cost).all())
            self.assertEqual(metadata['bandwidth'], bandwidth)
            json.dumps(metadata, allow_nan=False)


class SinkhornTests(unittest.TestCase):
    def test_low_cost_analytic_and_extreme_epsilon(self):
        cost = np.array([[0., 1.], [1., 0.]])
        for epsilon in (.05, 1.e-8, 1.e5):
            result = sinkhorn_log(cost, epsilon=epsilon)
            expected = 1 / (2 * (1 + np.exp(-1 / epsilon)))
            self.assertAlmostEqual(result['plan'][0, 0], expected)
            self.assertLessEqual(result['transport_cost'], result['independent_cost'] + 1e-12)
            self.assertTrue(result['converged'])

    def test_rectangular_weighted_marginals(self):
        cost = np.array([[0., .4, 1.], [.2, .3, 0.]])
        for a, b in ((None, None), ([.3, .7], [.2, .5, .3]), ([0., 1.], [.2, .5, .3])):
            result = sinkhorn_log(cost, a=a, b=b)
            np.testing.assert_allclose(result['plan'].sum(1), [.5, .5] if a is None else a, atol=1e-8)
            np.testing.assert_allclose(result['plan'].sum(0), [1/3]*3 if b is None else b, atol=1e-8)
            json.dumps({k: v for k, v in result.items() if k != 'plan'}, allow_nan=False)

    def test_nonconvergence_is_explicit(self):
        with self.assertRaises(SinkhornConvergenceError):
            sinkhorn_log([[0., .2], [.9, 0.]], a=[.2, .8], epsilon=.001, max_iter=1)
        for kwargs in ({'epsilon': 0}, {'max_iter': 0}, {'a': [.2, .2]}):
            with self.assertRaises(ValueError):
                sinkhorn_log([[0., 1.], [1., 0.]], **kwargs)


class SamplingTests(unittest.TestCase):
    def test_joint_sampling_distribution_seed_and_persistent_rng(self):
        plan = np.array([[.1, .2], [.3, .4]])
        i, j = draw_plan_pairs(plan, 100000, seed=7)
        hist = np.bincount(2*i+j, minlength=4) / len(i)
        np.testing.assert_allclose(hist, plan.ravel(), atol=.004)
        ii, jj = draw_plan_pairs(plan, 100000, seed=7)
        np.testing.assert_array_equal(i, ii)
        np.testing.assert_array_equal(j, jj)
        rng = np.random.default_rng(7)
        first = draw_plan_pairs(plan, 100, rng=rng)
        second = draw_plan_pairs(plan, 100, rng=rng)
        self.assertFalse(np.array_equal(first, second))
        with self.assertRaises(ValueError):
            draw_plan_pairs(plan, 2, seed=1, rng=rng)

    def test_complete_coupling_has_json_diagnostics(self):
        x = np.random.default_rng(1).normal(size=(4, 8, 1))
        result = couple_endpoints(x, x, count=20, seed=8)
        self.assertEqual(len(result['source_indices']), 20)
        self.assertEqual(result['cost'].shape, (4, 4))
        self.assertEqual(result['plan'].shape, (4, 4))
        json.dumps(result['diagnostics'], allow_nan=False)


if __name__ == '__main__':
    unittest.main()
