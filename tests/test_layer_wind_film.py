import unittest
import numpy as np
import torch
from lf_operator.conditioning import fit_condition_scalers,standardize_conditions,tensor_conditions,prepare_condition
from lf_operator.residual_flow import ConditionalVelocity,create_velocity_model

PROFILE=dict(wind_profile='power_law',height=112.,rotor_bottom_height=59.6,wind_reference_height=115.6,shear_exponent=.2)


class LayerWindFiLMTests(unittest.TestCase):
    def arrays(self,n=16):
        rng=np.random.default_rng(42);b=3
        return dict(case_ids=np.array(list('abc')),conditions=np.array([[12.,1.5],[16.,3.],[20.,5.]]),
            geometry=np.repeat(np.array([[[4.2,.18]],[[3.6,.13]],[[4.7,.18]]]),20,axis=1),
            z=np.repeat(((np.arange(20)+.5)/20)[None],b,axis=0),weights=np.ones((b,20)),
            theta=(np.arange(n)+.5)*2*np.pi/n,lf_prediction=rng.normal(size=(b,n,3)),
            rotor_features=rng.normal(size=(b,32,n)),field_features=rng.normal(size=(b,32,20,n)))

    def test_physical_profile_train_scaling_and_no_repeated_geometry(self):
        arrays=self.arrays();stats=fit_condition_scalers(arrays,[0,1],wind_profile=PROFILE)
        scaled=standardize_conditions(arrays,stats)
        expected=arrays['conditions'][:,0,None]*((59.6+112*arrays['z'])/115.6)**.2
        recovered=scaled['layer_wind']*np.asarray(stats['layer_wind']['scale'])+stats['layer_wind']['mean']
        np.testing.assert_allclose(recovered,expected,rtol=1e-6)
        self.assertAlmostEqual(np.asarray(stats['layer_wind']['mean']).item(),expected[:2].mean())
        self.assertEqual(scaled['scalar_geometry'].shape,(3,2))
        self.assertNotIn('geometry',scaled)
        self.assertEqual(stats['wind_profile'],PROFILE)
        arrays['geometry'][0,1,0]+=1
        with self.assertRaises(ValueError):standardize_conditions(arrays,stats)

    def test_zero_initialization_gradient_learning_reload_and_resolution(self):
        opts=dict(width=8,blocks=2,modes=6,direct_conditioning=True)
        torch.manual_seed(42);old=ConditionalVelocity(**opts)
        torch.manual_seed(42);model=create_velocity_model(**opts,architecture='layer-wind-film-path-v1')
        self.assertEqual(model.physical_encoder[0].in_features,24)
        for key,value in old.state_dict().items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        for n in (16,32):
            arrays=self.arrays(n);stats=fit_condition_scalers(arrays,[0,1],wind_profile=PROFILE)
            tensors=tensor_conditions(standardize_conditions(arrays,stats),arrays,'cpu')
            condition=prepare_condition(model,tensors,[0,1]);base=prepare_condition(old,tensors,[0,1])
            g=torch.randn(2,n,3);t=torch.tensor([.2,.8])
            torch.testing.assert_close(model(g,t,condition),old(g,t,base),rtol=0,atol=0)
        model(g,t,condition).square().mean().backward()
        self.assertGreater(model.physical_heads[0].weight.grad.abs().sum().item(),0.)
        torch.optim.SGD(model.parameters(),lr=.01).step()
        left=prepare_condition(model,tensors,[0,1]);tensors['layer_wind']+=1
        right=prepare_condition(model,tensors,[0,1])
        self.assertFalse(torch.equal(model(g,t,left),model(g,t,right)))
        restored=create_velocity_model(**model.config);restored.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(model(g,t,right),restored(g,t,right))


if __name__=='__main__':unittest.main()
