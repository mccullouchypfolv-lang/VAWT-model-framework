import unittest
from unittest.mock import patch
import numpy as np
from lf_operator import blade_pipeline as pipeline


class IndependentCouplingTests(unittest.TestCase):
    def test_independent_endpoints_are_sampled_without_any_ot(self):
        residual=np.random.default_rng(5).normal(size=(6,16,3))
        arrays=dict(residual=residual);splits=dict(train=[0,1,2,3],val=[4],test=[5])
        stats=pipeline.residual_normalization(residual,splits['train'])
        source=dict(channels=[dict(n_theta=16,n_channels=1,psd=(1/(1+np.arange(9)**2)).tolist())]*3)
        rng=np.random.default_rng(42)
        expected_ids=rng.choice(splits['train'],32,replace=True)
        expected_source=pipeline.sample_multichannel_source(source,32,rng)
        expected_target=pipeline.encode_residual(residual[expected_ids],stats)
        expected_t=rng.uniform(0,1,(32,1,1)).astype('float32')
        with patch('lf_operator.kernel_ot.couple_endpoints',side_effect=AssertionError('OT must not run')),patch(
                'lf_operator.joint_source_coupling.couple_joint_endpoints',side_effect=AssertionError('joint OT must not run')):
            ids,xt,t,u,d=pipeline.paired_batch(arrays,splits,stats,source,32,np.random.default_rng(42),dict(method='independent'))
        np.testing.assert_array_equal(ids,expected_ids)
        np.testing.assert_array_equal(t,expected_t)
        np.testing.assert_array_equal(u,expected_target-expected_source)
        np.testing.assert_array_equal(xt,(1-t)*expected_source+t*expected_target)
        self.assertEqual(d['method'],'independent')
        self.assertTrue(d['conditional_source_independence_by_construction'])
        self.assertNotIn('marginal_error',d)

    def test_invalid_and_conflicting_modes_fail_before_sampling(self):
        for pairing,joint in [(dict(method='typo'),None),(dict(method='independent',epsilon=.02),None),
                              (dict(method='independent'),dict(condition_weight=1.))]:
            with self.subTest(pairing=pairing,joint=joint),self.assertRaises(ValueError):
                pipeline.paired_batch({}, {}, {}, {},4,np.random.default_rng(42),pairing,joint_pairing=joint)


if __name__=='__main__':unittest.main()
