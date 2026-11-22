import unittest
import numpy as np
import torch
from lf_operator.blade_pipeline import validate_parameter_contract,lf_predict_parameters,TARGETS


class ParameterPipelineTests(unittest.TestCase):
    def test_same_lf_weights_accept_requested_angular_resolution(self):
        checkpoint=self.checkpoint()
        query=dict(wind_ref_m_s=9.,tsr_ref=5.,chord_m=3.2,thickness_ratio=.21)
        original={k:v.clone() for k,v in checkpoint['model'].items()}
        for n in [80,160,320,640]:
            with self.subTest(n=n):
                out=lf_predict_parameters(checkpoint,query,'cpu',n_theta=n)
                self.assertEqual(out['lf_prediction'].shape,(1,n,3))
                self.assertEqual(out['field_features'].shape,(1,4,20,n))
                np.testing.assert_allclose(out['theta'],(np.arange(n)+.5)*2*np.pi/n,atol=3e-7)
                self.assertTrue(np.isfinite(out['lf_prediction']).all())
        for k,value in checkpoint['model'].items():torch.testing.assert_close(value,original[k])
        self.assertEqual(checkpoint['source_config']['ntheta'],160)
        for n in [True,1.5,0,2,3]:
            with self.subTest(invalid=n),self.assertRaises(ValueError):
                lf_predict_parameters(checkpoint,query,'cpu',n_theta=n)

    def config(self):
        return dict(nz=20,ntheta=160,rotor_bottom_height=59.6,height=112.,radius=55.,rho=1.225)

    def checkpoint(self):
        from lf_operator.blade_data import INPUT_CONTRACT,FEATURES,CONDITIONS,TARGET_CONTRACT
        from lf_operator.model import FiLMFNO
        model=FiLMFNO(n_features=2,n_conditions=2,n_outputs=3,width=4,blocks=1,modes_z=2,modes_theta=3,pad_z=2)
        return dict(input_contract=INPUT_CONTRACT,features=list(FEATURES),conditions=list(CONDITIONS),targets=TARGETS,
                    target_contract=TARGET_CONTRACT,model_config=model.config,model=model.state_dict(),source_config=self.config(),
                    statistics=dict(x_mean=[3.,.2],x_std=[1.,.05],a_mean=[10.,3.],a_std=[3.,1.],y_mean=[0.,0.,0.],y_std=[1.,1.,1.]))

    def test_old_reconstruction_checkpoint_is_rejected(self):
        checkpoint=self.checkpoint()
        validate_parameter_contract(checkpoint)
        checkpoint.pop('input_contract')
        with self.assertRaisesRegex(ValueError,'input'):
            validate_parameter_contract(checkpoint)

    def test_wrong_target_and_condition_order_are_rejected(self):
        checkpoint=self.checkpoint()
        checkpoint['targets']=['Cp','Cq']
        with self.assertRaisesRegex(ValueError,'target'):validate_parameter_contract(checkpoint)
        checkpoint=self.checkpoint()
        checkpoint['conditions']=list(reversed(checkpoint['conditions']))
        with self.assertRaisesRegex(ValueError,'input'):validate_parameter_contract(checkpoint)

    def test_geometry_and_conditions_alone_are_sufficient_for_inference(self):
        checkpoint=self.checkpoint()
        query=dict(case_id='new',wind_ref_m_s=9.,tsr_ref=5.,chord_m=np.linspace(2,4,20).tolist(),thickness_ratio=np.linspace(.14,.21,20).tolist())
        output=lf_predict_parameters(checkpoint,[query],'cpu')
        self.assertEqual(output['lf_prediction'].shape,(1,160,3))
        self.assertEqual(output['conditions'].shape,(1,2))
        self.assertTrue(np.isfinite(output['lf_prediction']).all())
        self.assertEqual(output['field_features'].shape,(1,4,20,160))
        self.assertEqual(str(output['case_ids'][0]),'new')

    def test_corrected_two_condition_residual_flow_accepts_parameter_derived_features(self):
        from lf_operator.residual_flow import ConditionalVelocity,integrate_velocity
        from lf_operator.conditioning import tensor_conditions,prepare_condition
        checkpoint=self.checkpoint()
        query=dict(wind_ref_m_s=9.,tsr_ref=5.,chord_m=[3.2]*20,thickness_ratio=[.21]*20)
        arrays=lf_predict_parameters(checkpoint,[query],'cpu')
        tensors=tensor_conditions({k:arrays[k] for k in ('lf_prediction','conditions','rotor_features','field_features')},arrays,'cpu')
        model=ConditionalVelocity(width=4,blocks=1,modes=3,height_width=2,n_targets=3,lf_channels=3,n_conditions=2,field_channels=4,rotor_channels=4)
        condition=prepare_condition(model,tensors,[0])
        source=torch.randn(1,160,3)
        result=integrate_velocity(model,source,condition,steps=2)
        self.assertEqual(result.shape,(1,160,3))
        self.assertEqual(condition['modulation'].shape,(1,2))
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        self.assertIsNotNone(model.blocks[0].film[0].weight.grad)
        restored=ConditionalVelocity(**model.config)
        restored.load_state_dict(model.state_dict())
        self.assertEqual(restored.config['n_conditions'],2)


if __name__=='__main__':unittest.main()
