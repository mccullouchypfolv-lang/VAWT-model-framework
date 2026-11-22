import unittest
import numpy as np
from lf_operator.blade_pipeline import residual_normalization, encode_residual, decode_residual, sample_multichannel_source, paired_batch, draw_predictions
from lf_operator.blade_pipeline import validate_flow_assets


class BladePipelineTests(unittest.TestCase):
    def test_zero_velocity_evaluates_same_source_and_mean_on_new_grids(self):
        import torch
        from lf_operator.residual_flow import ConditionalVelocity
        from lf_operator.resolution import midpoint_theta,evaluate_periodic_profile
        from lf_operator.conditioning import tensor_conditions
        model=ConditionalVelocity(width=4,blocks=1,modes=3,height_width=2)
        for p in model.parameters():p.data.zero_()
        ref=midpoint_theta(16)
        means=np.stack([np.sin(ref),np.cos(2*ref),np.sin(3*ref)],axis=-1)
        stats={'mean_profile':means.tolist(),'scale':[1.,2.,3.]}
        channel={'n_theta':16,'n_channels':1,'psd':(1/(1+np.arange(9)**2)**2).tolist()}
        source={'channels':[channel]*3}
        native=sample_multichannel_source(source,4,np.random.default_rng(42))
        for n in [16,32,64]:
            theta=midpoint_theta(n)
            arrays=dict(case_ids=np.array(['a','b']),theta=theta,
                lf_prediction=np.ones((2,n,3)),conditions=np.zeros((2,2)),
                rotor_features=np.zeros((2,32,n)),field_features=np.zeros((2,32,20,n)),
                z=np.tile((np.arange(20)+.5)/20,(2,1)),weights=np.ones((2,20)))
            tensors=tensor_conditions({k:arrays[k] for k in ('lf_prediction','conditions','field_features','rotor_features')},arrays,'cpu')
            result=draw_predictions(model,tensors,arrays,[0,1],source,stats,2,2,42,reference_theta=ref)
            expected_noise=np.stack([evaluate_periodic_profile(x,ref,theta) for x in native]).reshape(2,2,n,3)
            expected_mean=np.stack([np.sin(theta),np.cos(2*theta),np.sin(3*theta)],axis=-1)
            np.testing.assert_allclose(result['samples'],1+expected_noise*np.array([1,2,3])+expected_mean,atol=2e-6)
            if n==16:
                old=draw_predictions(model,tensors,arrays,[0,1],source,stats,2,2,42)
                np.testing.assert_array_equal(result['samples'],old['samples'])

    def test_new_velocity_outputs_do_not_overwrite_previous_stage3(self):
        from lf_operator.blade_pipeline import flow_output_directory,ROOT
        cfg={'work':'artifacts/example','flow_output':'stage3_filmfno_v2'}
        self.assertEqual(flow_output_directory(cfg),ROOT/'artifacts/example/stage3_filmfno_v2')
        self.assertEqual(flow_output_directory({'work':'artifacts/example'}),ROOT/'artifacts/example/stage3')
        for name in ['..','../stage2','stage2','stage1','/tmp/other']:
            with self.subTest(name=name),self.assertRaises(ValueError):
                flow_output_directory(dict(cfg,flow_output=name))

    def test_previous_velocity_checkpoint_requires_retraining(self):
        from lf_operator.blade_pipeline import validate_velocity_checkpoint
        with self.assertRaisesRegex(ValueError,'retrain'):
            validate_velocity_checkpoint({'model_config':{'width':32,'blocks':3}})
        validate_velocity_checkpoint({'model_config':{'architecture':'floral-filmfno-v1'}})

    def test_three_independent_residual_channels_and_train_only_stats(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(7,16,3))*[1.,.01,.3]
        stats=residual_normalization(x,[0,1,2,3])
        z=encode_residual(x,stats)
        np.testing.assert_allclose(decode_residual(z,stats),x,atol=1e-6)
        np.testing.assert_allclose(np.mean(z[:4],axis=0),0,atol=1e-6)
        x[4:]+=1000
        self.assertEqual(stats,residual_normalization(x,[0,1,2,3]))

    def test_multichannel_source_reproducibility_and_no_duplicate_channels(self):
        channel={'n_theta':16,'n_channels':1,'psd':np.ones(9).tolist()}
        specs={'channels':[channel,channel,channel]}
        a=sample_multichannel_source(specs,6,np.random.default_rng(42))
        b=sample_multichannel_source(specs,6,np.random.default_rng(42))
        self.assertEqual(a.shape,(6,16,3))
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a[...,0],a[...,1]))

    def test_joint_ot_endpoints_follow_training_target_indices(self):
        rng=np.random.default_rng(42)
        arrays={'residual':rng.normal(size=(6,16,3))}
        splits={'train':[0,1,2,3],'val':[4],'test':[5]}
        stats=residual_normalization(arrays['residual'],splits['train'])
        channel={'n_theta':16,'n_channels':1,'psd':(1/(1+np.arange(9)**2)**2).tolist()}
        ids,xt,t,u,info=paired_batch(arrays,splits,stats,{'channels':[channel]*3},8,rng,
                                 {'kernel':'sobolev_rbf','epsilon':.1,'max_iter':5000,'tol':1e-8})
        self.assertTrue(set(ids)<=set(splits['train']))
        recovered_target=xt+(1-t)*u
        np.testing.assert_allclose(recovered_target,encode_residual(arrays['residual'][ids],stats),atol=1e-6)
        self.assertLessEqual(info['marginal_error'],1e-8)

    def test_flow_rejects_overlap_and_inconsistent_residuals(self):
        from lf_operator.blade_data import TARGET_CONTRACT,INPUT_CONTRACT
        residual=np.random.default_rng(4).normal(size=(6,16,3))
        arrays=dict(case_ids=np.array(list('abcdef')),geometry_ids=np.array(list('abcdef')),
                    target=residual,lf_prediction=np.zeros_like(residual),residual=residual.copy())
        splits={'train':[0,1,2,3],'val':[4],'test':[5]}
        stats=residual_normalization(residual,splits['train'])
        source=dict(channels=[{}, {}, {}],targets=stats['targets'],fit_indices=splits['train'],target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT)
        validate_flow_assets(arrays,splits,stats,source)
        with self.assertRaises(ValueError):validate_flow_assets(arrays,dict(splits,test=[4]),stats,source)
        arrays['residual'][0,0,0]+=1
        with self.assertRaises(ValueError):validate_flow_assets(arrays,splits,stats,source)

    def test_zero_velocity_ode_preserves_three_independent_source_channels(self):
        import torch
        from lf_operator.residual_flow import ConditionalVelocity
        from lf_operator.conditioning import tensor_conditions
        model=ConditionalVelocity(width=4,blocks=1,modes=3,height_width=2,n_targets=3,lf_channels=3)
        for parameter in model.parameters():parameter.data.zero_()
        arrays=dict(case_ids=np.array(['a','b']),theta=(np.arange(16)+.5)*2*np.pi/16,
                    lf_prediction=np.ones((2,16,3)),conditions=np.zeros((2,2)),
                    rotor_features=np.zeros((2,32,16)),field_features=np.zeros((2,32,20,16)),
                    z=np.tile((np.arange(20)+.5)/20,(2,1)),weights=np.ones((2,20)))
        tensors=tensor_conditions({k:arrays[k] for k in ('lf_prediction','conditions','field_features','rotor_features')},arrays,'cpu')
        channel={'n_theta':16,'n_channels':1,'psd':(1/(1+np.arange(9)**2)**2).tolist()}
        source={'channels':[channel]*3}
        stats={'mean_profile':np.zeros((16,3)).tolist(),'scale':[1.,2.,3.]}
        output=draw_predictions(model,tensors,arrays,[0,1],source,stats,4,2,42)
        noise=sample_multichannel_source(source,8,np.random.default_rng(42)).reshape(2,4,16,3)
        np.testing.assert_allclose(output['samples'],1+noise*np.array([1,2,3]),atol=1e-6)


if __name__=='__main__':unittest.main()
