import unittest
import torch
from lf_operator.residual_flow import ConditionalVelocity, create_velocity_model
from lf_operator.conditioning import prepare_condition


class PhysicalFiLMPathTests(unittest.TestCase):
    def inputs(self, n):
        return dict(lf_prediction=torch.randn(2,n,3), rotor_features=torch.randn(2,32,n),
                    field_features=torch.randn(2,32,20,n), conditions=torch.randn(2,2),
                    z=torch.linspace(0,1,20)[None].repeat(2,1), weights=torch.ones(2,20),
                    theta=((torch.arange(n)+.5)*2*torch.pi/n)[None].repeat(2,1),
                    geometry=torch.randn(2,20,2))

    def test_zero_initialization_preserves_default_and_heads_learn(self):
        opts=dict(width=8,blocks=2,modes=6,direct_conditioning=True)
        torch.manual_seed(42); old=ConditionalVelocity(**opts)
        torch.manual_seed(42); new=create_velocity_model(**opts,architecture='physical-film-path-v1')
        for key,value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[key],value,rtol=0,atol=0)
        tensors=self.inputs(16); g=torch.randn(2,16,3); t=torch.tensor([.2,.8])
        a=prepare_condition(old,tensors,[0,1]); b=prepare_condition(new,tensors,[0,1])
        torch.testing.assert_close(old(g,t,a),new(g,t,b),rtol=0,atol=0)
        new(g,t,b).square().mean().backward()
        for head in new.physical_heads:
            self.assertGreater(head.weight.grad.abs().sum().item(),0.)
        optimizer=torch.optim.SGD(new.parameters(),lr=.01);optimizer.step()
        left=prepare_condition(new,tensors,[0,1]);tensors['geometry']=tensors['geometry']+2
        right=prepare_condition(new,tensors,[0,1])
        self.assertFalse(torch.equal(new(g,t,left),new(g,t,right)))

    def test_resolution_reload_and_missing_geometry(self):
        model=create_velocity_model(width=8,blocks=2,modes=6,direct_conditioning=True,
                                    architecture='physical-film-path-v1')
        restored=create_velocity_model(**model.config);restored.load_state_dict(model.state_dict(),strict=True)
        for n in (16,32):
            tensors=self.inputs(n);g=torch.randn(2,n,3);t=torch.tensor([.2,.8])
            a=prepare_condition(model,tensors,[0,1]);b=prepare_condition(restored,tensors,[0,1])
            torch.testing.assert_close(model(g,t,a),restored(g,t,b))
        del tensors['geometry']
        with self.assertRaises(ValueError):prepare_condition(model,tensors,[0,1])


if __name__=='__main__':unittest.main()
