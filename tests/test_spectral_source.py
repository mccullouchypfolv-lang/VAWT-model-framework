"""Physical Fourier conventions and train-only spectral-source contracts."""
import json
import unittest

import numpy as np

from lf_operator import spectral_source as spectral


def training_curves(beta=3.8, n=160, cases=512):
    rng = np.random.default_rng(19)
    k = np.arange(n // 2 + 1, dtype=float)
    psd = np.ones_like(k)
    psd[1:] = k[1:] ** -beta
    z = (rng.normal(size=(cases, len(k))) +
         1j * rng.normal(size=(cases, len(k)))) * np.sqrt(psd / 2)
    z[:, 0] = rng.normal(size=cases) * .2 + 1.25
    if n % 2 == 0:
        z[:, -1] = rng.normal(size=cases) * np.sqrt(psd[-1])
    return (np.fft.irfft(z, n=n) * n)[..., None]


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = training_curves()
        cls.spec = spectral.fit_source(cls.train, bootstrap=60)

    def test_known_tail_and_admissibility(self):
        s = self.spec
        self.assertAlmostEqual(s['beta'], 3.8, delta=.15)
        self.assertGreater(s['beta_r_squared'], .98)
        # A finite-case percentile CI need not cover the population exponent;
        # arithmetic frequency binning also introduces a small slope bias.
        self.assertLess(s['beta_ci'][0], s['beta'])
        self.assertGreater(s['beta_ci'][1], s['beta'])
        self.assertGreater(s['beta_ci'][1] - s['beta_ci'][0], .001)
        self.assertAlmostEqual(s['nu_upper'], s['beta_ci'][0] / 2 - 1 - .05)
        self.assertGreater(s['nu'], 0)
        self.assertLess(s['nu'], s['nu_upper'])
        self.assertEqual(s['n_channels'], 1)
        self.assertEqual(s['tail_band'], [12, 60])
        json.dumps(s, allow_nan=False)

    def test_preserves_raw_moments_and_training_labels(self):
        original = self.train.copy()
        s = spectral.fit_source(self.train, bootstrap=20)
        z = np.fft.rfft(self.train[..., 0], axis=1, norm='forward')
        np.testing.assert_allclose(s['target_psd'], np.mean(abs(z-z.mean(0))**2, 0))
        np.testing.assert_allclose(s['power_psd'], np.mean(abs(z)**2, 0))
        np.testing.assert_array_equal(self.train, original)
        self.assertAlmostEqual(s['psd'][0], np.var(z[:, 0]).real)
        self.assertIn('covariance_offdiagonal_energy_fraction', s['diagnostics'])
        self.assertIn('mu', s['diagnostics'])
        self.assertIn('linear_bridge_L', s['diagnostics'])

    def test_sampler_psd_parseval_dc_and_nyquist(self):
        # A non-negligible Nyquist variance catches the common factor-two error.
        spec = {'n_theta': 16, 'n_channels': 1, 'psd': [.4, .2, .1, .3, .1, .2, .1, .1, .7]}
        x = spectral.sample_source(spec, 30000, seed=7)
        self.assertEqual(x.shape, (30000, 16, 1))
        self.assertTrue(np.isfinite(x).all())
        empirical = np.mean(abs(np.fft.rfft(x[..., 0], norm='forward'))**2, 0)
        np.testing.assert_allclose(empirical, spec['psd'], rtol=.035)
        expected = spec['psd'][0] + spec['psd'][-1] + 2 * sum(spec['psd'][1:-1])
        self.assertAlmostEqual(np.mean(x*x), expected, delta=.07)

    def test_reproducibility_rng_and_odd_grid(self):
        a = spectral.sample_source(self.spec, 4, seed=3)
        np.testing.assert_array_equal(a, spectral.sample_source(self.spec, 4, seed=3))
        self.assertFalse(np.array_equal(a, spectral.sample_source(self.spec, 4, seed=4)))
        np.testing.assert_array_equal(a, spectral.sample_source(self.spec, 4, rng=np.random.default_rng(3)))
        odd = {'n_theta': 15, 'n_channels': 1, 'psd': [1.] * 8}
        x = spectral.sample_source(odd, 20000, seed=4)
        self.assertEqual(x.shape, (20000, 15, 1))
        self.assertAlmostEqual(np.mean(x*x), 15, delta=.2)

    def test_controls_preserve_total_variance(self):
        def total(s):
            p = np.asarray(s['psd'])
            return p[0] + p[-1] + 2 * p[1:-1].sum()
        for s in (spectral.white_source(self.spec), spectral.fixed_matern_source(self.spec)):
            self.assertAlmostEqual(total(s), total(self.spec))
            self.assertEqual(s['n_theta'], 160)
        white = spectral.white_source(self.spec)
        np.testing.assert_allclose(white['psd'], white['psd'][0])
        fixed = spectral.fixed_matern_source(self.spec)
        self.assertEqual(fixed['nu'], .5)
        self.assertEqual(fixed['kappa'], 3.)

    def test_fit_rejects_complex_labels_instead_of_discarding_information(self):
        with self.assertRaises(ValueError):
            spectral.fit_source(self.train.astype(complex) + 1j, bootstrap=10)

    def test_invalid_and_empty_admissible_domains(self):
        with self.assertRaises(spectral.AdmissibilityError):
            spectral.fit_source(training_curves(beta=1.5), bootstrap=20)
        for x in (np.zeros((4, 160, 1)), np.zeros((1, 160, 1)), np.zeros((10, 160, 2))):
            with self.assertRaises(ValueError):
                spectral.fit_source(x)
        for kwargs in ({'tail_band': (70, 90)}, {'tail_band': (12, 14)}, {'bootstrap': 0}, {'confidence': 1.}):
            with self.assertRaises(ValueError):
                spectral.fit_source(self.train, **kwargs)
        for count in (-1, 0, 1.2, True):
            with self.assertRaises(ValueError):
                spectral.sample_source(self.spec, count)
        with self.assertRaises(ValueError):
            spectral.sample_source({'n_theta': 4, 'psd': [1., -1., 1.]}, 4)
        with self.assertRaises(ValueError):
            spectral.sample_source(self.spec, 2, seed=1, rng=np.random.default_rng(1))


if __name__ == '__main__':
    unittest.main()
