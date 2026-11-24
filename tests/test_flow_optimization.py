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
            cfg.update(flow_output='stage3_curve_aux',curve_auxiliary=dict(weight=.1,cases=2,draws=2,steps=2,method='heun'))
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()):
                blade_pipeline.train_flow(cfg)
            aux_history=json.loads((work/'stage3_curve_aux/history.json').read_text())
            for row in aux_history:
                self.assertTrue(np.isfinite(row['ode_curve_mse']))
                self.assertAlmostEqual(row['total_loss'],row['velocity_mse']+.1*row['ode_curve_mse'])
            restored=torch.load(work/'stage3_curve_aux/best.pt',weights_only=False)
            self.assertEqual(restored['config']['curve_auxiliary']['weight'],.1)
            cfg.update(flow_output='stage3_independent',pairing=dict(method='independent'))
            cfg.pop('curve_auxiliary')
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()),patch(
                    'lf_operator.kernel_ot.couple_endpoints',side_effect=AssertionError('independent training must not run OT')):
                report=blade_pipeline.train_flow(cfg)
            independent=json.loads((work/'stage3_independent/history.json').read_text())
            for row in independent:
                self.assertEqual(row['coupling_method'],'independent')
                self.assertNotIn('sinkhorn_marginal_error',row)
            self.assertTrue(report['conditional_source_independence_guaranteed'])
            ck=torch.load(work/'stage3_independent/best.pt',weights_only=False)
            self.assertEqual(ck['config']['pairing'],dict(method='independent'))
            self.assertEqual(ck['source'],source)
            cfg.update(flow_output='stage3_independent_timeweighted',flow_time_weighting='floral_quadratic')
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()):
                blade_pipeline.train_flow(cfg)
            weighted=json.loads((work/'stage3_independent_timeweighted/history.json').read_text())
            for row in weighted:
                self.assertEqual(row['time_weighting'],'floral_quadratic')
                self.assertGreaterEqual(row['time_weighted_velocity_mse'],row['velocity_mse']-1e-7)
                self.assertLessEqual(row['time_weighted_velocity_mse'],3*row['velocity_mse']+1e-7)
            ck=torch.load(work/'stage3_independent_timeweighted/best.pt',weights_only=False)
            self.assertEqual(ck['config']['flow_time_weighting'],'floral_quadratic')
            cfg.update(flow_output='stage3_aligned_fullpass',flow_epoch_mode='dataloader',
                flow_bridge=dict(method='floral',sig_min=1e-6,prior_covariance_factor=2.),
                flow_model=dict(width=4,blocks=2,modes=3,height_width=2,time_embedding_dim=10,
                    architecture='floral-filmfno-aligned-v2',direct_conditioning=True))
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()):
                blade_pipeline.train_flow(cfg)
            aligned=json.loads((work/'stage3_aligned_fullpass/history.json').read_text())
            for i,row in enumerate(aligned,1):
                self.assertEqual(row['epoch_samples'],4)
                self.assertEqual(row['epoch_unique_conditions'],4)
                self.assertEqual(row['updates_in_epoch'],1)
                self.assertEqual(row['optimizer_updates'],i)
                self.assertEqual(row['bridge_method'],'floral')
            ck=torch.load(work/'stage3_aligned_fullpass/best.pt',weights_only=False)
            self.assertEqual(ck['source'],source)
            self.assertEqual(ck['model_config']['architecture'],'floral-filmfno-aligned-v2')
            cfg.update(flow_output='stage3_fm_validation_frequency10',flow_epoch_mode='monte_carlo',
                flow_model=dict(width=4,blocks=1,modes=3,height_width=2),flow_epochs=11,
                flow_validation_metric='velocity_mse',flow_scheduler_frequency=10)
            cfg.pop('flow_bridge')
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()),patch.object(
                    blade_pipeline,'draw_predictions',wraps=blade_pipeline.draw_predictions) as ode_evaluation:
                report=blade_pipeline.train_flow(cfg)
            self.assertEqual(ode_evaluation.call_count,2)  # final val/test only, not early stopping
            fm_history=json.loads((work/'stage3_fm_validation_frequency10/history.json').read_text())
            for row in fm_history:
                self.assertIn('val_time_weighted_velocity_mse',row)
                self.assertNotIn('val_standardized_curve_mse',row)
                self.assertAlmostEqual(row['learning_rate'],.001*.99**((row['epoch']-1)//10))
                self.assertAlmostEqual(row['next_learning_rate'],.001*.99**(row['epoch']//10))
            best=min(fm_history,key=lambda row:row['val_time_weighted_velocity_mse'])
            self.assertEqual(report['best_epoch'],best['epoch'])
            ck=torch.load(work/'stage3_fm_validation_frequency10/best.pt',weights_only=False)
            self.assertEqual(ck['scheduler']['last_epoch'],ck['epoch']//10)
            cfg.update(flow_output='stage3_low_frequency_fm',flow_epochs=3,
                flow_low_frequency_fm=dict(energy_fraction=.95,energy_power=.5,strength=1.))
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()):
                report=blade_pipeline.train_flow(cfg)
            spectral_history=json.loads((work/'stage3_low_frequency_fm/history.json').read_text())
            for row in spectral_history:
                self.assertAlmostEqual(row['optimized_velocity_loss'],
                    row['time_weighted_velocity_mse']+row['low_frequency_fm_loss'],places=5)
                self.assertAlmostEqual(row['val_optimized_velocity_loss'],
                    row['val_time_weighted_velocity_mse']+row['val_low_frequency_fm_loss'],places=5)
            best=min(spectral_history,key=lambda row:row['val_optimized_velocity_loss'])
            self.assertEqual(report['best_epoch'],best['epoch'])
            ck=torch.load(work/'stage3_low_frequency_fm/best.pt',weights_only=False)
            fitted=json.loads((work/'stage3_low_frequency_fm/low_frequency_weights.json').read_text())
            self.assertEqual(ck['low_frequency_fm'],fitted)
            self.assertEqual(fitted['fit_indices'],splits['train'])
            self.assertEqual(ck['source'],source)
            cfg.update(flow_output='stage3_low_plus_complex',flow_complex_spectral=dict(strength=.1))
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()):
                report=blade_pipeline.train_flow(cfg)
            complex_history=json.loads((work/'stage3_low_plus_complex/history.json').read_text())
            for row in complex_history:
                self.assertAlmostEqual(row['optimized_velocity_loss'],row['time_weighted_velocity_mse']+
                    row['low_frequency_fm_loss']+.1*row['complex_spectral_l1'],places=5)
                self.assertAlmostEqual(row['val_optimized_velocity_loss'],row['val_time_weighted_velocity_mse']+
                    row['val_low_frequency_fm_loss']+.1*row['val_complex_spectral_l1'],places=5)
            self.assertEqual(report['best_epoch'],min(complex_history,key=lambda row:row['val_optimized_velocity_loss'])['epoch'])
            ck=torch.load(work/'stage3_low_plus_complex/best.pt',weights_only=False)
            self.assertEqual(ck['config']['flow_complex_spectral'],{'strength':.1})
            self.assertEqual(ck['source'],source)
            # Exercise real training, train-only geometry scaling and ODE evaluation.
            cfg.pop('flow_low_frequency_fm');cfg.pop('flow_complex_spectral')
            cfg.update(flow_output='stage3_physical_film_path',flow_epochs=2,
                flow_model=dict(width=4,blocks=1,modes=3,height_width=2,
                    architecture='physical-film-path-v1',direct_conditioning=True))
            geometry=rng.normal(size=(n,20,2))
            (work/'lf_cache').mkdir();(work/'lf_cache/manifest.json').write_text('{}',encoding='utf-8')
            def attach_geometry(values,*args):return dict(values,geometry=geometry)
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()),patch(
                    'lf_operator.conditioning.geometry_from_cache',side_effect=attach_geometry):
                report=blade_pipeline.train_flow(cfg)
            ck=torch.load(work/'stage3_physical_film_path/best.pt',weights_only=False)
            self.assertEqual(ck['source'],source)
            self.assertEqual(ck['splits'],splits)
            np.testing.assert_allclose(ck['scalers']['geometry']['mean'],geometry[:4].mean((0,1),keepdims=True))
            from lf_operator.residual_flow import create_velocity_model
            restored=create_velocity_model(**ck['model_config']);restored.load_state_dict(ck['model'],strict=True)
            self.assertGreater(restored.physical_heads[0].weight.abs().sum().item(),0.)
            self.assertTrue((work/'stage3_physical_film_path/test_predictions.npz').exists())
            cfg.update(flow_output='stage3_layer_wind_film_path',flow_velocity_loss='huber',flow_huber_delta=1.)
            cfg['flow_model']['architecture']='layer-wind-film-path-v1'
            geometry=np.repeat(np.abs(geometry[:,:1])+.1,20,axis=1)
            arrays['conditions']=np.abs(arrays['conditions'])+1
            np.savez(assets/'paired_data.npz',**arrays)
            profile=dict(wind_profile='power_law',height=112.,rotor_bottom_height=59.6,
                         wind_reference_height=115.6,shear_exponent=.2)
            (work/'lf_cache/manifest.json').write_text(json.dumps(dict(config=profile)),encoding='utf-8')
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()),patch(
                    'lf_operator.conditioning.geometry_from_cache',side_effect=attach_geometry):
                blade_pipeline.train_flow(cfg)
            ck=torch.load(work/'stage3_layer_wind_film_path/best.pt',weights_only=False)
            self.assertEqual(ck['source'],source);self.assertEqual(ck['splits'],splits)
            self.assertEqual(ck['scalers']['wind_profile'],profile)
            self.assertNotIn('geometry',ck['scalers'])
            restored=create_velocity_model(**ck['model_config']);restored.load_state_dict(ck['model'],strict=True)
            self.assertEqual(restored.physical_encoder[0].in_features,24)
            self.assertGreater(restored.physical_heads[0].weight.abs().sum().item(),0.)
            self.assertEqual(ck['config']['flow_velocity_loss'],'huber')
            rows=json.loads((work/'stage3_layer_wind_film_path/history.json').read_text())
            for row in rows:
                self.assertEqual(row['velocity_objective'],'huber')
                self.assertAlmostEqual(row['optimized_velocity_loss'],row['time_weighted_velocity_huber'])
                self.assertNotIn('time_weighted_velocity_mse',row)
            cfg.update(flow_output='stage3_angular_state')
            cfg.pop('flow_velocity_loss');cfg.pop('flow_huber_delta')
            cfg['flow_model']['architecture']='angular-state-film-v1'
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()),patch(
                    'lf_operator.conditioning.geometry_from_cache',side_effect=attach_geometry):
                blade_pipeline.train_flow(cfg)
            ck=torch.load(work/'stage3_angular_state/best.pt',weights_only=False)
            self.assertEqual(ck['source'],source);self.assertEqual(ck['splits'],splits)
            self.assertNotIn('geometry',ck['scalers']);self.assertNotIn('wind_profile',ck['scalers'])
            self.assertIn('scalar_geometry',ck['scalers'])
            restored=create_velocity_model(**ck['model_config']);restored.load_state_dict(ck['model'],strict=True)
            self.assertGreater(restored.state_to_condition.weight.abs().sum().item(),0.)
            self.assertTrue((work/'stage3_angular_state/test_predictions.npz').exists())
            from lf_operator.floral_source import floral_matern_source
            fixed_source=dict(channels=[floral_matern_source(nt) for _ in range(3)],fit_indices=splits['train'],
                targets=blade_pipeline.TARGETS,target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT,
                split_method='geometry',parameter_fit='fixed official FLORAL prior; no HF spectrum fitting')
            before=blade_pipeline.digest(assets/'source.json')
            cfg.update(flow_output='stage3_fixed_floral_source',flow_assets='stage2_fixed_floral',
                source_design=dict(method='floral_fixed_matern'),
                flow_model=dict(width=4,blocks=1,modes=3,height_width=2,direct_conditioning=True))
            with patch.object(blade_pipeline,'ROOT',scratch),contextlib.redirect_stdout(io.StringIO()),patch(
                    'lf_operator.spectral_source.fit_source',side_effect=AssertionError('fixed GP must not fit HF spectra')):
                blade_pipeline.prepare_residuals(cfg)
                blade_pipeline.train_flow(cfg)
            ck=torch.load(work/'stage3_fixed_floral_source/best.pt',weights_only=False)
            self.assertEqual(ck['source']['channels'],fixed_source['channels']);self.assertEqual(ck['splits'],splits)
            self.assertEqual(blade_pipeline.digest(assets/'source.json'),before)
            self.assertEqual(blade_pipeline.digest(work/'stage2_fixed_floral/normalization.json'),blade_pipeline.digest(assets/'normalization.json'))
            self.assertTrue((work/'stage3_fixed_floral_source/test_predictions.npz').exists())
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
