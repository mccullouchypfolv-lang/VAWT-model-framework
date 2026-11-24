import unittest
import numpy as np
from lf_operator.floral_source import floral_matern_source,floral_covariance
from lf_operator.spectral_source import sample_source
from lf_operator.resolution import evaluate_source,midpoint_theta


class FloralSourceTests(unittest.TestCase):
    def test_official_parameters_residual_covariance_and_euclidean_domain(self):
        spec=floral_matern_source(160)
        self.assertEqual(spec['nu'],.5);self.assertEqual(spec['lengthscale'],.001)
        self.assertEqual(spec['outputscale'],1.);self.assertEqual(spec['residual_covariance_factor'],2.)
        self.assertEqual(spec['confidence'],0.)
        theta=midpoint_theta(160);cov=floral_covariance(spec,theta)
        np.testing.assert_allclose(np.diag(cov),2.)
        self.assertAlmostEqual(cov[0,1],2*np.exp(-1/160/.001),places=12)
        self.assertLess(cov[0,-1],1e-100)  # official kernel is not circular
        self.assertNotIn('target_psd',spec);self.assertNotIn('psd',spec)

    def test_exact_sampler_covariance_reproducibility_and_new_resolution(self):
        spec=floral_matern_source(12,lengthscale=.1)
        draws=sample_source(spec,60000,seed=42)
        np.testing.assert_allclose(draws[:,:,0].mean(0),0.,atol=.02)
        np.testing.assert_allclose(np.cov(draws[:,:,0],rowvar=False,bias=True),
            floral_covariance(spec,midpoint_theta(12)),rtol=.04,atol=.035)
        a=sample_source(spec,4,seed=42)
        b=evaluate_source(spec,4,midpoint_theta(12),np.random.default_rng(42))
        np.testing.assert_array_equal(a,b)
        for n in (8,24,48):
            got=evaluate_source(spec,4,midpoint_theta(n),np.random.default_rng(42))
            self.assertEqual(got.shape,(4,n,1));self.assertTrue(np.isfinite(got).all())
        with self.assertRaises(ValueError):sample_source(spec,2,seed=42,rng=np.random.default_rng(42))
        for value in (0.,-1.,float('nan'),True):
            with self.subTest(value=value),self.assertRaises(ValueError):floral_matern_source(16,lengthscale=value)


if __name__=='__main__':unittest.main()
