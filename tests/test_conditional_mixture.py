import unittest
import torch
from lf_operator.conditional_mixture import ConditionalMixtureSource,expected_geodesic_loss,select_mixture_centers


class ConditionalMixtureTests(unittest.TestCase):
    def test_fourier_center_uses_physical_angle_and_shared_coefficients_across_grids(self):
        model=ConditionalMixtureSource(modes=5,components=2,width=8)
        for n in (16,32):
            theta=(torch.arange(n)+.5)*2*torch.pi/n
            geometry=torch.zeros(2,20,2);z=torch.linspace(0,1,20).repeat(2,1);a=torch.zeros(2,2)
            lf=torch.cos(theta)[None,:,None].expand(2,-1,3)
            with torch.no_grad():
                model.centers.weight.zero_();model.centers.bias.zero_()
                coefficients=model.centers.bias.reshape(2,3,11)
                coefficients[:,:,0]=2;coefficients[:,:,1]=3;coefficients[:,:,6]=4
            means,probabilities=model(geometry,z,a,lf,theta[None].expand(2,-1))
            expected=2+3*torch.cos(theta)+4*torch.sin(theta)
            torch.testing.assert_close(means,expected[None,None,:,None].expand(2,2,-1,3))
            torch.testing.assert_close(probabilities.sum(1),torch.ones(2))

    def test_expected_cost_and_gradients_reach_centers_and_probabilities(self):
        centers=torch.tensor([0.,2.]).reshape(1,2,1,1).expand(1,2,8,3).clone().requires_grad_()
        logits=torch.tensor([[0.,0.]],requires_grad=True);target=torch.zeros(1,8,3)
        loss=expected_geodesic_loss(centers,logits.softmax(-1),target,noise_variance=.5)
        self.assertAlmostEqual(float(loss),2.5)
        loss.backward();self.assertGreater(float(centers.grad.abs().sum()),0)
        self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_descriptor_and_center_match_on_shared_nodes_of_different_grids(self):
        torch.manual_seed(42);model=ConditionalMixtureSource(modes=5,components=2,width=8)
        geometry=torch.randn(2,20,2);z=torch.linspace(0,1,20).repeat(2,1);operating=torch.randn(2,2)
        def evaluate(n):
            theta=torch.pi/16+torch.arange(n)*2*torch.pi/n
            lf=(torch.cos(theta)+.2*torch.sin(3*theta))[None,:,None].expand(2,-1,3)
            return model(geometry,z,operating,lf,theta[None].expand(2,-1))
        a,p=evaluate(16);b,q=evaluate(32)
        torch.testing.assert_close(a,b[:,:,::2],rtol=1e-4,atol=1e-7)
        torch.testing.assert_close(p,q)

    def test_categorical_selection_and_detached_velocity_path(self):
        means=torch.tensor([1.,5.]).reshape(1,2,1,1).expand(3,2,8,3).clone().requires_grad_()
        probs=torch.tensor([[.25,.75]]).expand(3,-1)
        selected,indices=select_mixture_centers(means,probs,torch.tensor([.1,.3,.99]))
        torch.testing.assert_close(indices,torch.tensor([0,1,1]))
        torch.testing.assert_close(selected[:,0,0],torch.tensor([1.,5.,5.]))
        with torch.no_grad():source,_=select_mixture_centers(means,probs,torch.tensor([.1,.3,.99]))
        velocity=torch.nn.Parameter(torch.tensor(1.));(source*velocity).square().mean().backward()
        self.assertIsNone(means.grad);self.assertIsNotNone(velocity.grad)

    def test_training_and_joint_checkpoint_sampling_roundtrip_preserve_frozen_assets(self):
        import contextlib,io,json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        from lf_operator import blade_pipeline as bp
        from lf_operator.mixture_pipeline import train_mixture_source,draw_mixture_predictions
        from lf_operator.residual_flow import create_velocity_model
        from lf_operator.conditioning import standardize_conditions,tensor_conditions
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';assets=work/'stage2';assets.mkdir(parents=True)
            (work/'stage1').mkdir();(work/'stage1/best.pt').write_bytes(b'frozen LF')
            rng=np.random.default_rng(7);n=6;nt=16
            arrays=dict(case_ids=np.array(list('abcdef')),geometry_ids=np.array(list('abcdef')),
                theta=(np.arange(nt)+.5)*2*np.pi/nt,conditions=rng.uniform(1.,3.,size=(n,2)),
                lf_prediction=np.zeros((n,nt,3)),rotor_features=rng.normal(size=(n,32,nt)),
                field_features=rng.normal(size=(n,32,20,nt)),z=np.tile(np.arange(20)/20,(n,1)),
                weights=np.ones((n,20)),geometry=rng.normal(size=(n,20,2)))
            arrays['target']=rng.normal(size=(n,nt,3));arrays['residual']=arrays['target'].copy()
            split=dict(train=[0,1,2,3],val=[4],test=[5]);stats=bp.residual_normalization(arrays['residual'],split['train'])
            source=dict(channels=[dict(n_theta=nt,n_channels=1,psd=[.01]*9)]*3,
                fit_indices=split['train'],targets=bp.TARGETS,target_contract=bp.TARGET_CONTRACT,input_contract=bp.INPUT_CONTRACT)
            np.savez(assets/'paired_data.npz',**arrays)
            for name,value in dict(status={'status':'complete'},provenance={},splits=split,normalization=stats,source=source).items():
                (assets/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
            identities={p:bp.digest(p) for p in (work/'stage1/best.pt',assets/'paired_data.npz',assets/'source.json')}
            cfg=dict(work='work',device='cpu',seed=42,flow_model=dict(width=4,blocks=1,modes=3,height_width=2),
                flow_lr=.001,flow_weight_decay=.0001,flow_epochs=2,flow_patience=90,flow_min_delta=.0001,validate_every=1,
                batch_size=2,updates_per_epoch=2,pairing={'method':'independent'},flow_time_weighting='floral_quadratic',
                flow_output='mixture',source_model=dict(components=2,modes=3,width=8),
                source_training=dict(warmup_epochs=0,cooldown_epochs=0,update_every=1,lr=.001,weight_decay=.0001),
                validation_draws=2,evaluation_draws=2,ode_steps=2,ode_method='euler')
            with patch.object(bp,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):report=train_mixture_source(cfg)
            history=json.loads((work/'mixture/history.json').read_text())
            self.assertEqual(history[-1]['source_optimizer_updates'],4)
            self.assertTrue(all(r['source_phase']=='alternating' for r in history))
            self.assertEqual(report['best_epoch'],min(history,key=lambda r:r['val_standardized_curve_mse'])['epoch'])
            ck=torch.load(work/'mixture/best.pt',weights_only=False)
            self.assertEqual(ck['inference_mode'],'conditional_mixture_ode');self.assertEqual(ck['source'],source)
            self.assertEqual(ck['scalers']['fit_indices'],split['train'])
            v=create_velocity_model(**ck['model_config']).eval();v.load_state_dict(ck['model'])
            base=ConditionalMixtureSource(**ck['source_model_config']).eval();base.load_state_dict(ck['source_model'])
            tensors=tensor_conditions(standardize_conditions(arrays,ck['scalers']),arrays,'cpu')
            a=draw_mixture_predictions(v,base,tensors,arrays,[5],source,stats,2,2,20042,method='euler')
            b=draw_mixture_predictions(v,base,tensors,arrays,[5],source,stats,2,2,20042,method='euler')
            with np.load(work/'mixture/test_predictions.npz') as saved:
                np.testing.assert_array_equal(a['mean'],saved['mean'])
            np.testing.assert_array_equal(a['samples'],b['samples'])
            for p,h in identities.items():self.assertEqual(bp.digest(p),h)
            with self.assertRaisesRegex(ValueError,'conditional_mixture_ode'):bp.validate_velocity_checkpoint(ck)


if __name__=='__main__':unittest.main()
