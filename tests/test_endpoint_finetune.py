import unittest
import numpy as np
import torch
from lf_operator import endpoint_finetune as endpoint


class EndpointFineTuneTests(unittest.TestCase):
    def test_midpoint_uses_zero_state_and_one_call_with_gradients(self):
        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bias=torch.nn.Parameter(torch.tensor([1.,2.,3.]));self.calls=0
            def forward(self,state,time,condition):
                self.calls+=1
                self.state=state.detach().clone();self.time=time.detach().clone()
                return state+self.bias
        model=Recorder();condition={'features':torch.ones(2,7,16)}
        result=endpoint.midpoint_residual(model,condition,3)
        self.assertEqual(model.calls,1)
        self.assertEqual(result.shape,(2,16,3))
        self.assertEqual(float(model.state.abs().max()),0)
        torch.testing.assert_close(model.time,torch.full_like(model.time,.5))
        result.sum().backward()
        torch.testing.assert_close(model.bias.grad,torch.full((3,),32.))

    def test_physical_endpoint_decode_and_channelwise_relative_loss(self):
        state=torch.zeros(2,16,3,requires_grad=True)
        lf=torch.full_like(state,2.)
        stats={'mean_profile':np.ones((16,3)).tolist(),'scale':[2.,.2,.02]}
        prediction=endpoint.compose_endpoint(state,lf,stats)
        torch.testing.assert_close(prediction,torch.full_like(state,3.))
        target=torch.full_like(state,4.)
        loss=endpoint.endpoint_relative_l2(prediction,target)
        self.assertAlmostEqual(float(loss),.25,places=6)
        multiplier=torch.tensor([100.,.01,.0001])
        scaled=endpoint.endpoint_relative_l2(prediction*multiplier,target*multiplier)
        torch.testing.assert_close(loss,scaled,atol=1e-6,rtol=1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue((state.grad<0).all())

    def test_deterministic_checkpoint_cannot_be_loaded_as_an_ode_sampler(self):
        from lf_operator.blade_pipeline import validate_velocity_checkpoint
        with self.assertRaisesRegex(ValueError,'deterministic'):
            validate_velocity_checkpoint({'model_config':{'architecture':'floral-filmfno-v1'},
                'inference_mode':'deterministic_midpoint'})

    def test_pretrained_endpoint_training_preserves_assets_and_selects_validation(self):
        import contextlib,io,json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from lf_operator import blade_pipeline as bp
        from lf_operator.residual_flow import create_velocity_model
        from lf_operator.conditioning import fit_condition_scalers
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';assets=work/'stage2';assets.mkdir(parents=True)
            (work/'stage1').mkdir();(work/'stage1/best.pt').write_bytes(b'frozen LF')
            parent_dir=work/'parent';parent_dir.mkdir()
            rng=np.random.default_rng(7);n=6;nt=16
            arrays=dict(case_ids=np.array(list('abcdef')),geometry_ids=np.array(list('abcdef')),
                theta=(np.arange(nt)+.5)*2*np.pi/nt,conditions=rng.uniform(1.,3.,size=(n,2)),
                lf_prediction=np.zeros((n,nt,3)),rotor_features=rng.normal(size=(n,32,nt)),
                field_features=rng.normal(size=(n,32,20,nt)),z=np.tile(np.arange(20)/20,(n,1)),weights=np.ones((n,20)))
            arrays['target']=rng.normal(size=(n,nt,3));arrays['residual']=arrays['target'].copy()
            splits=dict(train=[0,1,2,3],val=[4],test=[5]);stats=bp.residual_normalization(arrays['residual'],splits['train'])
            source=dict(channels=[dict(n_theta=nt,n_channels=1,psd=np.ones(nt//2+1).tolist())]*3,
                fit_indices=splits['train'],targets=bp.TARGETS,target_contract=bp.TARGET_CONTRACT,input_contract=bp.INPUT_CONTRACT)
            np.savez(assets/'paired_data.npz',**arrays)
            for name,value in dict(status={'status':'complete'},provenance={},splits=splits,normalization=stats,source=source).items():
                (assets/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
            model_cfg=dict(width=4,blocks=1,modes=3,height_width=2)
            model=create_velocity_model(**model_cfg,n_targets=3,lf_channels=3,n_conditions=2)
            parent_path=parent_dir/'best.pt'
            torch.save(dict(model=model.state_dict(),model_config=model.config,config={'flow_model':model_cfg},
                epoch=85,scalers=fit_condition_scalers(arrays,splits['train']),splits=splits,residual_stats=stats,
                source=source,lf_checkpoint_sha256=bp.digest(work/'stage1/best.pt'),
                input_contract=bp.INPUT_CONTRACT,target_contract=bp.TARGET_CONTRACT),parent_path)
            identities={p:bp.digest(p) for p in [parent_path,assets/'paired_data.npz',assets/'source.json']}
            cfg=dict(work='work',device='cpu',seed=42,flow_model=model_cfg,flow_lr=.0002,flow_weight_decay=.0001,
                flow_epochs=2,flow_patience=90,flow_min_delta=.0001,validate_every=1,batch_size=2,
                updates_per_epoch=2,flow_output='endpoint',endpoint_parent=dict(checkpoint='work/parent/best.pt',sha256=bp.digest(parent_path)))
            with patch.object(bp,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                report=endpoint.train_endpoint(cfg)
            history=json.loads((work/'endpoint/history.json').read_text())
            self.assertEqual(history[0]['epoch'],0)
            best=min(history,key=lambda row:row['val_endpoint_relative_l2'])
            self.assertEqual(report['best_epoch'],best['epoch'])
            self.assertEqual(report['epochs_run'],2)
            ck=torch.load(work/'endpoint/best.pt',weights_only=False)
            self.assertEqual(ck['inference_mode'],'deterministic_midpoint')
            self.assertEqual(ck['source'],source)
            self.assertEqual(ck['scalers']['fit_indices'],splits['train'])
            for path,identity in identities.items():self.assertEqual(bp.digest(path),identity)
            with np.load(work/'endpoint/test_predictions.npz') as d:
                np.testing.assert_array_equal(d['target'],arrays['target'][[5]])
                self.assertEqual(d['mean'].shape,(1,nt,3))
            cfg['flow_output']='invalid_parent';cfg['endpoint_parent']['sha256']='incorrect'
            with patch.object(bp,'ROOT',root),self.assertRaisesRegex(ValueError,'identity'):
                endpoint.train_endpoint(cfg)
            self.assertFalse((work/'invalid_parent').exists())


if __name__=='__main__':unittest.main()
