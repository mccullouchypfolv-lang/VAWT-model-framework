import unittest
import numpy as np
import torch
from lf_operator.residual_flow import ConditionalVelocity,create_velocity_model
from lf_operator.conditioning import fit_condition_scalers,standardize_conditions,tensor_conditions,prepare_condition
from lf_operator.angular_state_velocity import periodic_relaxation


class AngularStateTests(unittest.TestCase):
    def test_periodic_ode_analytic_response_resolution_and_gradient(self):
        tau=torch.tensor([.2,1.],dtype=torch.float64,requires_grad=True)
        for n in (16,32,160):
            theta=(torch.arange(n,dtype=torch.float64)+.5)*2*torch.pi/n
            u=theta.cos()[None,None].repeat(1,2,1)
            got=periodic_relaxation(u,tau)
            expected=(theta.cos()[None,None]+tau[None,:,None]*theta.sin()[None,None])/(1+tau[None,:,None]**2)
            torch.testing.assert_close(got,expected,rtol=1e-10,atol=1e-10)
            torch.testing.assert_close(periodic_relaxation(u.roll(3,-1),tau),got.roll(3,-1))
            torch.testing.assert_close(periodic_relaxation(torch.ones_like(u)*2,tau),torch.ones_like(u)*2)
        got.square().mean().backward();self.assertGreater(tau.grad.abs().sum().item(),0.)

    def arrays(self,n):
        rng=np.random.default_rng(42)
        return dict(case_ids=np.array(list('abc')),conditions=rng.normal(size=(3,2)),
            geometry=np.repeat(np.array([[[4.2,.18]],[[3.6,.13]],[[4.7,.18]]]),20,axis=1),
            z=np.tile((np.arange(20)+.5)/20,(3,1)),weights=np.ones((3,20)),theta=(np.arange(n)+.5)*2*np.pi/n,
            lf_prediction=rng.normal(size=(3,n,3)),rotor_features=rng.normal(size=(3,32,n)),field_features=rng.normal(size=(3,32,20,n)))

    def test_initial_identity_state_training_and_reload(self):
        opts=dict(width=8,blocks=2,modes=6,direct_conditioning=True)
        torch.manual_seed(42);old=ConditionalVelocity(**opts)
        torch.manual_seed(42);model=create_velocity_model(**opts,architecture='angular-state-film-v1')
        for key,value in old.state_dict().items():
            actual=model.state_dict()[key]
            if key=='lift.0.weight':actual=actual[:,:value.shape[1]]
            torch.testing.assert_close(actual,value,rtol=0,atol=0)
        for n in (16,32):
            arrays=self.arrays(n);stats=fit_condition_scalers(arrays,[0,1],include_scalar_geometry=True)
            tensors=tensor_conditions(standardize_conditions(arrays,stats),arrays,'cpu')
            self.assertNotIn('geometry',tensors);self.assertEqual(tensors['scalar_geometry'].shape,(3,2))
            condition=prepare_condition(model,tensors,[0,1]);base=prepare_condition(old,tensors,[0,1])
            self.assertEqual(condition['angular_state'].shape,(2,8,n))
            g=torch.randn(2,n,3);t=torch.tensor([.2,.8])
            torch.testing.assert_close(model(g,t,condition),old(g,t,base),rtol=0,atol=0)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        model(g,t,condition).square().mean().backward()
        self.assertGreater(model.state_to_condition.weight.grad.abs().sum().item(),0.)
        self.assertGreater(model.lift[0].weight.grad[:,-8:].abs().sum().item(),0.)
        optimizer.step();optimizer.zero_grad()
        condition=prepare_condition(model,tensors,[0,1]);model(g,t,condition).square().mean().backward()
        self.assertGreater(model.state_encoder.raw_tau.grad.abs().sum().item(),0.)
        left=prepare_condition(model,tensors,[0,1]);tensors['scalar_geometry']+=1
        right=prepare_condition(model,tensors,[0,1])
        self.assertFalse(torch.equal(model(g,t,left),model(g,t,right)))
        restored=create_velocity_model(**model.config);restored.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(model(g,t,right),restored(g,t,right))


if __name__=='__main__':unittest.main()
