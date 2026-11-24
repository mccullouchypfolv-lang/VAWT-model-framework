import unittest
import torch
from lf_operator.residual_flow import ConditionalVelocity,create_velocity_model
from lf_operator.target_decoupled_flow import TargetDecoupledVelocity


class TargetDecoupledTests(unittest.TestCase):
    def condition(self,model,n=16):
        return model.prepare_condition(torch.randn(2,n,3),torch.randn(2,32,n),torch.randn(2,32,20,n),
            torch.randn(2,2),torch.linspace(0,1,20)[None].repeat(2,1),torch.ones(2,20),
            torch.arange(n).float()[None].repeat(2,1)*2*torch.pi/n)

    def test_same_seed_initial_function_matches_baseline(self):
        options=dict(width=8,blocks=2,modes=6,direct_conditioning=True,lift_ratio=2,projection_ratio=2)
        torch.manual_seed(42);base=ConditionalVelocity(**options)
        torch.manual_seed(42);model=TargetDecoupledVelocity(**options)
        c=self.condition(base);g=torch.randn(2,16,3);t=torch.tensor([.2,.8])
        torch.testing.assert_close(model(g,t,c),base(g,t,c),rtol=1e-5,atol=1e-6)
        self.assertEqual(len(model.heads),3)
        for left,right in [(model.heads[0][0].weight,model.heads[1][0].weight),
                           (model.blocks[0].films[0][0].weight,model.blocks[0].films[1][0].weight)]:
            self.assertNotEqual(left.data_ptr(),right.data_ptr())

    def test_single_target_gradient_does_not_update_other_target_modules(self):
        model=TargetDecoupledVelocity(width=8,blocks=2,modes=6,direct_conditioning=True)
        c=self.condition(model);p=model(torch.randn(2,16,3),torch.tensor([.2,.8]),c)
        p[:,:,0].square().mean().backward()
        for j in [1,2]:
            modules=[model.heads[j],model.initial_films[j],model.linear_films[j]]+[b.films[j] for b in model.blocks]
            for module in modules:
                for parameter in module.parameters():
                    self.assertTrue(parameter.grad is None or torch.count_nonzero(parameter.grad)==0)
        self.assertGreater(float(model.heads[0][-1].weight.grad.abs().sum()),0)

    def test_factory_checkpoint_and_variable_resolution(self):
        model=TargetDecoupledVelocity(width=8,blocks=1,modes=6,direct_conditioning=True)
        restored=create_velocity_model(**model.config);restored.load_state_dict(model.state_dict(),strict=True)
        for n in [16,32]:
            c=self.condition(model,n);g=torch.randn(2,n,3);t=torch.tensor([.3,.7])
            torch.testing.assert_close(model(g,t,c),restored(g,t,c))


if __name__=='__main__':unittest.main()
