import unittest
import torch
from lf_operator.residual_flow import ConditionalVelocity
from lf_operator.conditioning import prepare_condition


class ScalarGeometryFiLMTests(unittest.TestCase):
    def test_common_weights_preserved_and_only_condition_input_expanded(self):
        options=dict(width=8,blocks=1,modes=6,direct_conditioning=True)
        torch.manual_seed(42);old=ConditionalVelocity(**options)
        torch.manual_seed(42);new=ConditionalVelocity(**options,scalar_geometry_film=True)
        self.assertEqual(new.condition_encoder[0].in_channels,old.condition_encoder[0].in_channels+2)
        self.assertEqual(new.lift[0].in_channels,old.lift[0].in_channels)
        for key,value in old.state_dict().items():
            got=new.state_dict()[key]
            if key=='condition_encoder.0.weight':got=got[:,:value.shape[1]]
            torch.testing.assert_close(got,value,rtol=0,atol=0)

    def test_geometry_affects_film_but_not_direct_features(self):
        torch.manual_seed(42);model=ConditionalVelocity(width=8,blocks=1,modes=6,direct_conditioning=True,scalar_geometry_film=True)
        n=16
        tensors=dict(lf_prediction=torch.randn(2,n,3),rotor_features=torch.randn(2,32,n),
            field_features=torch.randn(2,32,20,n),conditions=torch.randn(2,2),z=torch.linspace(0,1,20)[None].repeat(2,1),
            weights=torch.ones(2,20),theta=(torch.arange(n).float()+.5)[None].repeat(2,1)*2*torch.pi/n,
            geometry=torch.zeros(2,20,2))
        left=prepare_condition(model,tensors,[0,1]);tensors['geometry']=torch.ones(2,20,2)
        right=prepare_condition(model,tensors,[0,1])
        torch.testing.assert_close(left['features'],right['features'],rtol=0,atol=0)
        self.assertFalse(torch.equal(left['encoded_features'],right['encoded_features']))
        pred=model(torch.randn(2,n,3),torch.tensor([.2,.7]),right);pred.square().mean().backward()
        self.assertGreater(float(model.condition_encoder[0].weight.grad[:,-2:].abs().sum()),0)
        restored=ConditionalVelocity(**model.config);restored.load_state_dict(model.state_dict(),strict=True)

    def test_nonuniform_geometry_is_not_silently_pooled(self):
        model=ConditionalVelocity(width=8,blocks=1,modes=6,scalar_geometry_film=True)
        geometry=torch.zeros(1,20,2);geometry[:,1]=1
        with self.assertRaises(ValueError):
            model.prepare_condition(torch.zeros(1,16,3),torch.zeros(1,32,16),torch.zeros(1,32,20,16),
                torch.zeros(1,2),torch.zeros(1,20),torch.ones(1,20),torch.zeros(1,16),geometry=geometry)


if __name__=='__main__':unittest.main()
