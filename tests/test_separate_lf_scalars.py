import unittest
import numpy as np
import torch
from lf_operator.model import create_lf_model
from lf_operator.training_data import compact_lf_inputs,forward

class SeparateLFScalarsTests(unittest.TestCase):
    def test_independent_inputs_gradients_and_checkpoint(self):
        m=create_lf_model(n_features=1,n_conditions=4,width=4,blocks=1,modes_z=2,modes_theta=3,pad_z=2,input_layout='separate-scalar-layer-wind-v1')
        b=dict(x=torch.randn(2,20),theta=((torch.arange(16)+.5)*2*torch.pi/16).repeat(2,1))
        for key in m.scalar_names:b[key]=torch.randn(2,1,requires_grad=True)
        out=forward(m,b,features=True)
        self.assertEqual(out['prediction'].shape,(2,16,3))
        out['prediction'].square().mean().backward()
        for key in m.scalar_names:
            self.assertTrue(torch.isfinite(b[key].grad).all())
            self.assertGreater(float(b[key].grad.abs().sum()),0)
        clone=create_lf_model(**m.config);clone.load_state_dict(m.state_dict())
        torch.testing.assert_close(forward(m,b),forward(clone,b))
        torch.testing.assert_close(out['rotor_features'],out['field_features'].mean(2))
        bad=dict(b,wind_ref_m_s=torch.ones(2))
        with self.assertRaises(ValueError):forward(m,bad)

    def test_data_adapter_returns_four_scalars(self):
        x=np.broadcast_to(np.array([4.2,.18])[None,:,None],(20,2,16)).copy()
        cfg=dict(wind_profile='power_law',wind_reference_height=115.6,shear_exponent=.2)
        stats=dict(x_mean=[4.,.15],x_std=[1.,.03],a_mean=[10.,3.],a_std=[2.,1.])
        b=compact_lf_inputs(x,[12,2],np.linspace(60,170,20),np.arange(16),stats,cfg,separate=True)
        self.assertNotIn('a',b)
        for key,value in zip(['wind_ref_m_s','tsr_ref','chord_m','thickness_ratio'],[1.,-1.,.2,1.]):
            self.assertEqual(b[key].shape,(1,))
            np.testing.assert_allclose(b[key],[value],atol=1e-6)

    def test_parameter_inference_preserves_separate_checkpoint(self):
        from tests.test_parameter_pipeline import ParameterPipelineTests
        from lf_operator.blade_pipeline import lf_predict_parameters
        ck=ParameterPipelineTests().checkpoint()
        ck['source_config'].update(wind_profile='power_law',wind_reference_height=115.6,shear_exponent=.2)
        m=create_lf_model(n_features=1,n_conditions=4,width=4,blocks=1,modes_z=2,modes_theta=3,pad_z=2,input_layout='separate-scalar-layer-wind-v1')
        ck.update(model_config=m.config,model=m.state_dict(),features=['uninduced_wind_m_s'],conditions=list(m.scalar_names))
        for n in [80,160,320]:
            out=lf_predict_parameters(ck,dict(wind_ref_m_s=12.,tsr_ref=2.,chord_m=4.2,thickness_ratio=.18),'cpu',n_theta=n)
            self.assertEqual(out['lf_prediction'].shape,(1,n,3))
            self.assertTrue(np.isfinite(out['lf_prediction']).all())

    def test_film_is_applied_after_completed_block(self):
        from torch.nn import functional as F
        m=create_lf_model(n_features=1,n_conditions=4,width=4,blocks=1,modes_z=2,modes_theta=3,pad_z=2,input_layout='separate-scalar-layer-wind-v1')
        self.assertFalse(hasattr(m.blocks[0],'scalar_film'))
        b=dict(x=torch.randn(2,20),theta=((torch.arange(16)+.5)*2*torch.pi/16).repeat(2,1))
        for key in m.scalar_names:b[key]=torch.randn(2,1)
        captured={}
        hook=m.blocks[0].register_forward_hook(lambda module,args,out:captured.update(block=out))
        out=forward(m,b,features=True);hook.remove()
        parameters=sum(m.modulations[0].scalar_film[key](b[key]) for key in m.scalar_names)
        gamma,beta=parameters.chunk(2,dim=-1)
        expected=(1+gamma[:,:,None,None])*captured['block']+beta[:,:,None,None]
        torch.testing.assert_close(out['field_features'],expected)

    def test_official_grid_embedding_appends_only_angular_coordinate(self):
        m=create_lf_model(n_features=1,n_conditions=4,width=4,blocks=1,modes_z=2,modes_theta=3,pad_z=2,input_layout='separate-scalar-layer-wind-v1')
        self.assertEqual(m.grid_embedding.dim,1)
        self.assertEqual(m.lift.in_channels,2)
        for n in [16,32]:
            b=dict(x=torch.randn(2,20),theta=((torch.arange(n)+.5)*2*torch.pi/n).repeat(2,1))
            for key in m.scalar_names:b[key]=torch.randn(2,1)
            captured={}
            hook=m.lift.register_forward_pre_hook(lambda module,args:captured.update(input=args[0]))
            forward(m,b);hook.remove()
            torch.testing.assert_close(captured['input'][:,0],b['x'][:,:,None].expand(-1,-1,n))
            expected=(torch.arange(n)/n)[None,None,:].expand(2,20,-1)
            torch.testing.assert_close(captured['input'][:,1],expected)
        with self.assertRaises(ValueError):forward(m,dict(b,theta=b['theta']+.1))
