import copy,unittest
from unittest.mock import patch
import numpy as np
from lf_operator import joint_source_coupling as joint
from lf_operator.kernel_ot import couple_endpoints


class JointSourceCouplingTests(unittest.TestCase):
    def test_condition_penalty_and_product_mixture(self):
        dependent=np.eye(4)/4;labels=np.array([0,0,1,1]);product=np.full((4,4),1/16)
        self.assertAlmostEqual(joint.conditional_chi2(dependent,labels),1.)
        self.assertAlmostEqual(joint.conditional_chi2(product,labels),0.)
        self.assertAlmostEqual(joint.conditional_chi2(.6*dependent+.4*product,labels),.36)
        cost=1-np.eye(4)
        pi,d=joint.regularize_coupling(cost,dependent,labels,epsilon=.1,condition_weight=100.)
        np.testing.assert_allclose(pi.sum(0),.25,atol=1e-8)
        np.testing.assert_allclose(pi.sum(1),.25,atol=1e-8)
        self.assertGreater(d['independent_fraction'],0)
        self.assertLess(d['condition_chi2_after'],d['condition_chi2_before'])
        self.assertLessEqual(d['joint_objective'],min(d['ot_endpoint_objective'],d['independent_endpoint_objective'])+1e-9)

    def test_zero_conditional_weight_reproduces_existing_pairing(self):
        rng=np.random.default_rng(2);g=rng.normal(size=(8,16,3));r=rng.normal(size=(8,16,3))
        kwargs=dict(kernel='sobolev_rbf',epsilon=.1,max_iter=5000,tol=1e-6)
        old=couple_endpoints(g,r,seed=42,**kwargs)
        new=joint.couple_joint_endpoints(g,r,np.arange(8),seed=42,condition_weight=0,**kwargs)
        np.testing.assert_array_equal(old['source_indices'],new['source_indices'])
        np.testing.assert_array_equal(old['target_indices'],new['target_indices'])
        np.testing.assert_array_equal(old['plan'],new['plan'])
        self.assertEqual(new['diagnostics']['independent_fraction'],0.)

    def source(self):
        k=np.arange(9);psd=.05*(1+k*k)**(-.8)
        c=dict(n_theta=16,n_channels=1,nu=.3,nu_upper=.5,kappa=1.,amplitude=.05,
            psd=psd.tolist(),target_psd=(psd*1.2).tolist(),resolved_modes=6)
        return dict(channels=[copy.deepcopy(c) for _ in range(3)],fit_indices=[0,1,2,3])

    def test_candidates_preserve_variance_admissibility_and_original(self):
        source=self.source();original=copy.deepcopy(source)
        candidates=joint.source_candidates(source,nu_factors=[.5,1.],kappa_factors=[1.,2.])
        self.assertEqual(candidates[0]['source'],source)
        w=np.array([1]+[2]*7+[1])
        for candidate in candidates:
            for a,b in zip(source['channels'],candidate['source']['channels']):
                self.assertLess(b['nu'],b['nu_upper'])
                self.assertAlmostEqual(w@a['psd'],w@b['psd'],places=10)
                self.assertEqual(a['psd'][0],b['psd'][0])
        self.assertEqual(source,original)

    def test_selection_uses_only_training_residuals(self):
        from lf_operator.blade_pipeline import residual_normalization
        source=self.source();a=dict(residual=np.random.default_rng(3).normal(size=(6,16,3)),coupling_condition_ids=np.arange(6))
        splits=dict(train=[0,1,2,3],val=[4],test=[5]);stats=residual_normalization(a['residual'],splits['train'])
        cfg=dict(selection_batches=2,batch_size=8,nu_factors=[.5,1.],kappa_factors=[1.,2.],
                 spectral_weight=1.,transport_weight=1.,condition_weight=1.)
        pairing=dict(kernel='sobolev_rbf',epsilon=.1,max_iter=5000,tol=1e-6)
        with patch.object(joint,'kernel_cost',wraps=joint.kernel_cost) as measured:
            fitted,report=joint.select_joint_source(a,splits,stats,source,cfg,pairing,seed=42)
        for call in measured.call_args_list[1:]:
            self.assertEqual(call.kwargs['bandwidth'],report['fixed_metric']['bandwidth'])
            self.assertFalse(call.kwargs['median_normalize'])
        a['residual'][4:]=1e9
        other,again=joint.select_joint_source(a,splits,stats,source,cfg,dict(pairing,method='kernel_ot'),seed=42)
        self.assertEqual(fitted,other);self.assertEqual(report,again)
        self.assertEqual(report['fit_indices'],splits['train'])

    def test_physical_condition_ids_ignore_responses_and_group_equal_inputs(self):
        arrays=dict(conditions=np.array([[8.,2.],[8.,2.],[8.,3.]]),geometry=np.ones((3,20,2)),
                    target=np.zeros((3,16,3)))
        ids=joint.condition_identities(arrays)
        self.assertEqual(ids[0],ids[1]);self.assertNotEqual(ids[1],ids[2])
        arrays['target'][:]=1e9
        np.testing.assert_array_equal(ids,joint.condition_identities(arrays))
        arrays['geometry'][1,19,1]=.2
        self.assertEqual(len(set(joint.condition_identities(arrays))),3)

    def test_pipeline_keeps_targets_and_conditions_aligned(self):
        from lf_operator.blade_pipeline import paired_batch,residual_normalization,encode_residual
        source=self.source();a=dict(residual=np.random.default_rng(3).normal(size=(6,16,3)),coupling_condition_ids=np.arange(6))
        splits=dict(train=[0,1,2,3],val=[4],test=[5]);stats=residual_normalization(a['residual'],splits['train'])
        kwargs=dict(kernel='sobolev_rbf',epsilon=.1,max_iter=5000,tol=1e-6)
        old=paired_batch(a,splits,stats,source,8,np.random.default_rng(42),kwargs)
        same=paired_batch(a,splits,stats,source,8,np.random.default_rng(42),kwargs,
                          joint_pairing=dict(condition_weight=0.,transport_weight=1.))
        for expected,actual in zip(old[:4],same[:4]):np.testing.assert_array_equal(expected,actual)
        ids,xt,t,u,d=paired_batch(a,splits,stats,source,8,np.random.default_rng(42),kwargs,
                                 joint_pairing=dict(condition_weight=1.,transport_weight=1.))
        self.assertTrue(set(ids)<=set(splits['train']))
        np.testing.assert_allclose(xt+(1-t)*u,encode_residual(a['residual'][ids],stats),atol=3e-7)
        self.assertLessEqual(d['condition_chi2_after'],d['condition_chi2_before'])


if __name__=='__main__':unittest.main()
