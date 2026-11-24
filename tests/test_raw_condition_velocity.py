import unittest
import torch

from lf_operator.raw_condition_velocity import RawConditionVelocity


class RawConditionTests(unittest.TestCase):
    def prepare(self,model,n=16):
        return model.prepare_condition(None,None,None,torch.tensor([[.2,.3]]),
            torch.linspace(0,1,20)[None],torch.ones(1,20),
            (torch.arange(n).float()+.5)[None]*2*torch.pi/n,
            geometry=torch.randn(1,20,2))

    def test_complete_geometry_is_in_condition_not_state(self):
        model=RawConditionVelocity(width=8,blocks=2,modes=6,lift_ratio=2,projection_ratio=2)
        condition=self.prepare(model)
        self.assertEqual(condition['raw_condition'].shape,(1,64,16))
        self.assertEqual(model.lift[0].in_channels,3+32)
        self.assertEqual(model.condition_encoder[0].in_channels,64)
        self.assertEqual(model.blocks[0].film[0].in_channels,8)
        # Geometry/high level operating channels are broadcast; no height pooling.
        torch.testing.assert_close(condition['raw_condition'][:,:62,:1].expand(-1,-1,16),condition['raw_condition'][:,:62])
        out=model(torch.randn(1,16,3),torch.tensor([.5]),condition)
        out.square().mean().backward()
        self.assertGreater(float(model.condition_encoder[0].weight.grad.abs().sum()),0)

    def test_no_lf_dependency_and_checkpoint_resolution_transfer(self):
        torch.manual_seed(42);model=RawConditionVelocity(width=8,blocks=1,modes=6)
        geometry=torch.randn(1,20,2);a=torch.randn(1,2);z=torch.linspace(0,1,20)[None]
        for n in [16,32]:
            theta=(torch.arange(n).float()+.5)[None]*2*torch.pi/n
            left=model.prepare_condition(None,None,None,a,z,None,theta,geometry=geometry)
            right=model.prepare_condition(torch.full((1,n,3),float('nan')),None,None,a,z,None,theta,geometry=geometry)
            state=torch.randn(1,n,3);t=torch.tensor([.3])
            torch.testing.assert_close(model(state,t,left),model(state,t,right),rtol=0,atol=0)
            restored=RawConditionVelocity(**model.config);restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(state,t,left),restored(state,t,left))


if __name__=='__main__':unittest.main()
