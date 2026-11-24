import unittest
import numpy as np
import torch
from lf_operator.training_data import compact_lf_inputs, forward
from lf_operator.model import FiLMFNO

class CompactLFWindTests(unittest.TestCase):
    def test_inputs_and_prescribed_wind(self):
        cfg=dict(height=112.,rotor_bottom_height=59.6,wind_reference_height=115.6,shear_exponent=.2,wind_profile='power_law')
        stats=dict(x_mean=[4.,.15],x_std=[1.,.03],a_mean=[10.,3.],a_std=[2.,1.])
        x=np.broadcast_to(np.array([4.2,.18])[None,:,None],(20,2,16)).copy()
        z=59.6+(np.arange(20)+.5)*112/20
        b=compact_lf_inputs(x,np.array([12.,2.]),z,np.arange(16)*2*np.pi/16,stats,cfg)
        self.assertEqual(set(b),{'x','a','theta'})
        self.assertEqual(b['x'].shape,(20,))
        np.testing.assert_allclose(b['a'],[1.,-1.,.2,1.],atol=1e-6)
        expected=12*(z/115.6)**.2
        np.testing.assert_allclose(b['x']*2+10,expected,rtol=1e-6)
        x[3,0,0]+=1
        with self.assertRaises(ValueError):compact_lf_inputs(x,np.array([12.,2.]),z,b['theta'],stats,cfg)

    def test_model_mean_pool_backward_reload_resolution(self):
        m=FiLMFNO(n_features=1,n_conditions=4,width=8,blocks=1,modes_z=2,modes_theta=3,pad_z=2,input_layout='scalar-geometry-layer-wind-v1')
        for n in [16,32]:
            b=dict(x=torch.randn(2,20),a=torch.randn(2,4),theta=(torch.arange(n)*2*torch.pi/n).repeat(2,1))
            self.assertEqual(m.lift.in_channels,3)
            out=forward(m,b,features=True)
            torch.testing.assert_close(out['rotor_features'],out['field_features'].mean(2))
            self.assertEqual(out['prediction'].shape,(2,n,3))
            out['prediction'].square().mean().backward()
            copy=FiLMFNO(**m.config);copy.load_state_dict(m.state_dict())
            torch.testing.assert_close(forward(m,b),forward(copy,b))
        self.assertTrue(torch.isfinite(m.lift.weight.grad).all())

    def test_checkpoint_parameter_queries_and_cache_loader(self):
        import tempfile
        from pathlib import Path
        from tests.test_parameter_pipeline import ParameterPipelineTests
        from lf_operator.blade_pipeline import lf_predict_parameters,validate_parameter_contract
        from lf_operator.training_data import LFData
        ck=ParameterPipelineTests().checkpoint()
        ck['source_config'].update(wind_profile='power_law',wind_reference_height=115.6,shear_exponent=.2)
        m=FiLMFNO(n_features=1,n_conditions=4,width=4,blocks=1,modes_z=2,modes_theta=3,pad_z=2,input_layout='scalar-geometry-layer-wind-v1')
        ck.update(model_config=m.config,model=m.state_dict(),features=['uninduced_wind_m_s'],conditions=['wind_ref_m_s','tsr_ref','chord_m','thickness_ratio'])
        query=dict(wind_ref_m_s=12.,tsr_ref=2.,chord_m=4.2,thickness_ratio=.18)
        for n in (80,160,320):
            out=lf_predict_parameters(ck,query,'cpu',n_theta=n)
            self.assertEqual(out['lf_prediction'].shape,(1,n,3))
            self.assertTrue(np.isfinite(out['lf_prediction']).all())
        from lf_operator.blade_data import parameter_inputs,INPUT_CONTRACT,FEATURES,CONDITIONS
        raw=parameter_inputs(12.,2.,4.2,.18,ck['source_config'])
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory)/'case.npz',**raw)
            manifest=dict(config=ck['source_config'],records=[dict(file='case.npz')],input_contract=INPUT_CONTRACT,features=list(FEATURES),conditions=list(CONDITIONS))
            validate_parameter_contract(ck,manifest)
            batch=LFData(directory,manifest,[0],ck['statistics'],require_targets=False,input_layout=m.config['input_layout'])[0]
            self.assertEqual(set(batch),{'x','a','theta'})
            result=forward(m,{k:v[None] for k,v in batch.items()},features=True)
            reference=lf_predict_parameters(ck,query,'cpu')
            np.testing.assert_allclose(result['prediction'].detach().numpy(),reference['lf_prediction'],atol=1e-6)
        with self.assertRaises(ValueError):
            lf_predict_parameters(ck,dict(query,chord_m=np.linspace(3,4,20)),'cpu')

    def test_fresh_fixed_source_uses_new_lf_residuals_without_nago(self):
        import tempfile,json,contextlib,io
        from pathlib import Path
        from unittest.mock import patch
        from lf_operator import blade_pipeline as bp
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);work=root/'work';(work/'lf_cache').mkdir(parents=True);(work/'stage1').mkdir()
            records=[dict(case_id=str(i),geometry_id=str(i)) for i in range(10)]
            bp.save_json(work/'lf_cache/manifest.json',dict(records=records))
            torch.save(dict(manifest_sha256=bp.digest(work/'lf_cache/manifest.json'),splits={'train':[0]}),work/'stage1/best.pt')
            bp.save_json(work/'hf_provenance.json',{})
            theta=(np.arange(16)+.5)*2*np.pi/16
            rng=np.random.default_rng(42);target=rng.normal(size=(10,16,3));lf=rng.normal(size=target.shape)
            ids=np.array([str(i) for i in range(10)])
            np.savez(work/'hf_data.npz',target=target,case_ids=ids,geometry_ids=ids,theta=theta)
            arrays=dict(case_ids=ids,lf_prediction=lf,theta=theta)
            cfg=dict(work='work',seed=42,flow_assets='fixed',source_design={'method':'floral_fixed_matern'})
            with patch.object(bp,'ROOT',root),patch.object(bp,'setup',return_value='cpu'),patch.object(bp,'lf_predict',return_value=arrays),patch('lf_operator.spectral_source.fit_source',side_effect=AssertionError('must not fit NAGO')),contextlib.redirect_stdout(io.StringIO()):
                report=bp.prepare_residuals(cfg)
            self.assertEqual(report['status'],'complete')
            with np.load(work/'fixed/paired_data.npz') as data:np.testing.assert_allclose(data['residual'],target-lf)
            self.assertFalse(bp.read_json(work/'fixed/source.json')['nago_spectrum_fit'])
