import unittest
import torch
from lf_operator import blade_pipeline


class FlowOptimizationTests(unittest.TestCase):
    def test_training_loop_logs_epoch_rates_and_saves_scheduler(self):
        import contextlib,io,json,shutil,uuid
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        from lf_operator.blade_data import TARGET_CONTRACT,INPUT_CONTRACT
        root=Path(__file__).resolve().parents[1]
        scratch=root/'tmp'/('flow-scheduler-test-'+uuid.uuid4().hex)
        scratch.mkdir(parents=True)
        try:
            work=scratch/'work';assets=work/'stage2';assets.mkdir(parents=True)
            (work/'stage1').mkdir();(work/'stage1/best.pt').write_bytes(b'test LF identity')
            rng=np.random.default_rng(42);n=6;nt=16
            arrays=dict(case_ids=np.array(list('abcdef')),geometry_ids=np.array(list('abcdef')),
                theta=(np.arange(nt)+.5)*2*np.pi/nt,conditions=rng.normal(size=(n,2)),
                lf_prediction=np.zeros((n,nt,3)),rotor_features=rng.normal(size=(n,32,nt)),
                field_features=rng.normal(size=(n,32,20,nt)),z=np.tile(np.arange(20)/20,(n,1)),weights=np.ones((n,20)))
            arrays['target']=rng.normal(size=(n,nt,3));arrays['residual']=arrays['target'].copy()
            splits=dict(train=[0,1,2,3],val=[4],test=[5])
            stats=blade_pipeline.residual_normalization(arrays['residual'],splits['train'])
            source=dict(channels=[dict(n_theta=nt,n_channels=1,psd=(1/(1+np.arange(9)**2)).tolist())]*3,
                fit_indices=splits['train'],targets=blade_pipeline.TARGETS,target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT)
            np.savez(assets/'paired_data.npz',**arrays)
            for name,value in dict(status={'status':'complete'},provenance={},splits=splits,normalization=stats,source=source).items():
                (assets/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
            cfg=dict(work='work',device='cpu',seed=42,flow_model=dict(width=4,blocks=1,modes=3,height_width=2),
                flow_lr=.001,flow_weight_decay=1e-4,flow_lr_gamma=.99,flow_patience=90,flow_epochs=3,
                updates_per_epoch=2,batch_size=4,pairing=dict(kernel='sobolev_rbf',epsilon=.1,max_iter=5000,tol=1e-6),
                validate_every=1,validation_draws=2,evaluation_draws=2,ode_steps=2,ode_method='euler')
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()):
                blade_pipeline.train_flow(cfg)
            history=json.loads((work/'stage3/history.json').read_text())
            for i,row in enumerate(history):
                self.assertAlmostEqual(row['learning_rate'],.001*.99**i)
                self.assertAlmostEqual(row['next_learning_rate'],.001*.99**(i+1))
            ck=torch.load(work/'stage3/best.pt',weights_only=False)
            self.assertEqual(ck['scheduler']['last_epoch'],ck['epoch'])
            self.assertEqual(ck['optimizer']['param_groups'][0]['weight_decay'],1e-4)
        finally:
            self.assertTrue(scratch.resolve().is_relative_to((root/'tmp').resolve()))
            shutil.rmtree(scratch)

    def test_exponential_decay_applies_once_after_all_epoch_updates(self):
        parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer,scheduler=blade_pipeline.flow_optimization([parameter],dict(
            flow_lr=1e-3,flow_weight_decay=1e-4,flow_lr_gamma=.99))
        self.assertIsInstance(optimizer,torch.optim.AdamW)
        self.assertIsInstance(scheduler,torch.optim.lr_scheduler.ExponentialLR)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'],1e-4)
        for epoch in range(3):
            expected=1e-3*.99**epoch
            for _ in range(2):
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'],expected)
                optimizer.zero_grad();parameter.square().sum().backward();optimizer.step()
            scheduler.step()
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],expected*.99)
        self.assertLess(parameter.item(),1.)

    def test_old_configs_keep_constant_learning_rate_and_original_weight_decay(self):
        parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer,scheduler=blade_pipeline.flow_optimization([parameter],dict(flow_lr=1e-3))
        self.assertEqual(optimizer.param_groups[0]['weight_decay'],.001)
        parameter.sum().backward();optimizer.step();scheduler.step()
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-3)


if __name__=='__main__':unittest.main()
