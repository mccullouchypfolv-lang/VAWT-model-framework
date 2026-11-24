import unittest
import numpy as np

from lf_operator.pointwise_dc import pointwise_normalization,compress_source_dc
from lf_operator.blade_pipeline import encode_residual,decode_residual


class PointwiseDCTests(unittest.TestCase):
    def test_train_only_local_scale_and_roundtrip(self):
        rng=np.random.default_rng(1);r=rng.normal(size=(5,16,3));r[:,8:]*=3
        stats=pointwise_normalization(r,[0,1,2,3],floor_ratio=.25)
        changed=r.copy();changed[4]=1e6
        other=pointwise_normalization(changed,[0,1,2,3],floor_ratio=.25)
        np.testing.assert_array_equal(stats['scale'],other['scale'])
        scale=np.array(stats['scale']);self.assertEqual(scale.shape,(16,3))
        self.assertTrue(np.all(scale>=.25*np.array(stats['validation_scale'])))
        np.testing.assert_allclose(decode_residual(encode_residual(r,stats),stats),r,rtol=1e-6,atol=1e-6)

    def test_dc_only_compression_updates_diagnostics_without_mutation(self):
        from lf_operator.spectral_source import fit_source,sample_source
        # Use a simple prefit spectrum to test the change independent of fitting.
        spec=dict(n_theta=8,n_channels=1,psd=[4.,1.,.5,.25,.1],
            target_psd=[2.,1.,.5,.25,.1],diagnostics=dict(target_psd_floor=1e-12))
        changed=compress_source_dc(spec,.1)
        self.assertEqual(spec['psd'][0],4.)
        self.assertAlmostEqual(changed['psd'][0],.4)
        self.assertEqual(changed['psd'][1:],spec['psd'][1:])
        self.assertAlmostEqual(changed['diagnostics']['mu'][0],5.)
        draws=sample_source(changed,20000,seed=42)
        self.assertAlmostEqual(float(draws.mean(1).var()),.4,delta=.02)

    def test_invalid_hyperparameters(self):
        with self.assertRaises(ValueError):pointwise_normalization(np.ones((3,8,3)),[0,1],floor_ratio=0.)
        with self.assertRaises(ValueError):compress_source_dc({},0.)


if __name__=='__main__':unittest.main()
