import unittest
import numpy as np
import torch
from lf_operator import blade_pipeline as bp


class LinearVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__();self.rate=torch.nn.Parameter(torch.tensor(.7));self.config={}
    def prepare_condition(self,lf,rotor,field,a,z,weights,theta):return {'operating':a}
    def forward(self,g,t,condition):return self.rate*g


class CurveAuxiliaryTests(unittest.TestCase):
    def data(self):
        n,nt=5,16
        arrays=dict(residual=np.zeros((n,nt,3),dtype='float32'),theta=np.arange(nt)*2*np.pi/nt)
        tensors={k:torch.zeros(n,1) for k in ['lf_prediction','rotor_features','field_features','conditions','z','weights','theta']}
        splits=dict(train=[0,1,2],val=[3],test=[4])
        source=dict(channels=[dict(n_theta=nt,n_channels=1,psd=np.ones(9).tolist())]*3)
        stats=dict(mean_profile=np.zeros((nt,3)).tolist(),scale=[1.,2.,.1])
        return arrays,tensors,splits,source,stats

    def test_real_ode_endpoint_and_gradient_not_single_velocity_shortcut(self):
        arrays,tensors,splits,source,stats=self.data();m=LinearVelocity()
        loss,ids=bp.curve_endpoint_loss(m,tensors,arrays,splits,source,stats,np.random.default_rng(9),cases=2,draws=2,steps=4,method='heun')
        rng=np.random.default_rng(9);expected_ids=rng.choice(splits['train'],2,replace=False)
        g0=bp.sample_multichannel_source(source,4,rng)
        rate=float(m.rate.detach());factor=(1+rate/4+.5*(rate/4)**2)**4
        expected=np.mean((g0.reshape(2,2,16,3).mean(1)*factor)**2)
        self.assertAlmostEqual(float(loss.detach()),float(expected),places=5)
        self.assertFalse(np.isclose(float(loss.detach()),float(np.mean((g0.reshape(2,2,16,3).mean(1)*(1+rate))**2)),rtol=1e-3))
        np.testing.assert_array_equal(ids,expected_ids)
        loss.backward();self.assertGreater(abs(m.rate.grad.item()),0)
        results=[]
        for delta in [-1e-3,1e-3]:
            with torch.no_grad():m.rate.fill_(rate+delta)
            other,_=bp.curve_endpoint_loss(m,tensors,arrays,splits,source,stats,np.random.default_rng(9),cases=2,draws=2,steps=4,method='heun')
            results.append(float(other.detach()))
        np.testing.assert_allclose(m.rate.grad.item(),(results[1]-results[0])/.002,rtol=1e-4,atol=1e-4)

    def test_auxiliary_never_uses_holdout_labels_and_balances_channel_scales(self):
        arrays,tensors,splits,source,stats=self.data();m=LinearVelocity()
        first,ids=bp.curve_endpoint_loss(m,tensors,arrays,splits,source,stats,np.random.default_rng(42),cases=3,draws=2,steps=2)
        arrays['residual'][3:]=1e9
        again,_=bp.curve_endpoint_loss(m,tensors,arrays,splits,source,stats,np.random.default_rng(42),cases=3,draws=2,steps=2)
        torch.testing.assert_close(first,again)
        self.assertEqual(set(ids),set(splits['train']))
        arrays['residual'][:3]=np.array([1.,.2,.01])
        balanced,_=bp.curve_endpoint_loss(m,tensors,arrays,splits,source,stats,np.random.default_rng(42),cases=3,draws=2,steps=2)
        factor=np.array([10.,.01,100.])
        scaled=dict(arrays,residual=arrays['residual']*factor)
        scaled_stats=dict(stats,scale=(np.asarray(stats['scale'])*factor).tolist())
        same,_=bp.curve_endpoint_loss(m,tensors,scaled,splits,source,scaled_stats,np.random.default_rng(42),cases=3,draws=2,steps=2)
        torch.testing.assert_close(balanced,same)
        for count in [0,True,1.2]:
            with self.assertRaises(ValueError):bp.curve_endpoint_loss(m,tensors,arrays,splits,source,stats,np.random.default_rng(42),cases=count)


if __name__=='__main__':unittest.main()
