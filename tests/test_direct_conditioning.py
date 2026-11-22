"""Isolate LF/height concatenation from the existing FiLM information path."""
import json
import math
import unittest

import torch
from lf_operator.residual_flow import ConditionalVelocity


class DirectConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def condition(self,model,n=32):
        return model.prepare_condition(torch.randn(2,n,3),torch.randn(2,32,n),
            torch.randn(2,32,20,n),torch.randn(2,2),
            torch.linspace(0,1,20).expand(2,-1),torch.ones(2,20),
            (torch.arange(n)*2*math.pi/n).expand(2,-1))

    def test_direct_route_changes_only_nonlinear_branch_when_film_condition_is_fixed(self):
        for direct in [False,True]:
            model=ConditionalVelocity(width=8,blocks=1,modes=5,direct_conditioning=direct)
            c=self.condition(model)
            # Keep the encoded FiLM field fixed to isolate the added route.
            other=dict(c,features=c['features'].clone())
            other['features'][:,:3]+=3
            other['features'][:,35:43]+=2
            g=torch.randn(2,32,3);t=torch.tensor([.2,.7])
            linear,nonlinear=model.forward_branches(g,t,c)
            changed_linear,changed_nonlinear=model.forward_branches(g,t,other)
            torch.testing.assert_close(changed_linear,linear)
            if direct:self.assertGreater((nonlinear-changed_nonlinear).abs().max().item(),1e-5)
            else:torch.testing.assert_close(changed_nonlinear,nonlinear)

    def test_direct_route_gradients_checkpoint_and_angular_sizes(self):
        model=ConditionalVelocity(width=8,blocks=1,modes=5,direct_conditioning=True)
        for n in [32,64,128]:
            c=self.condition(model,n)
            c['encoded_features']=c['encoded_features'].detach()
            c['features']=c['features'].detach().requires_grad_()
            g=torch.randn(2,n,3);t=torch.tensor([.2,.7])
            linear,nonlinear=model.forward_branches(g,t,c)
            nonlinear.square().mean().backward()
            gradient=c['features'].grad
            self.assertGreater(gradient[:,:3].abs().sum().item(),0)
            self.assertGreater(gradient[:,35:43].abs().sum().item(),0)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(nonlinear.shape,g.shape)
            clone=ConditionalVelocity(**json.loads(json.dumps(model.config)))
            clone.load_state_dict(model.state_dict(),strict=True)
            torch.testing.assert_close(clone(g,t,c),linear+nonlinear)

    def test_default_keeps_previous_input_shape_and_requires_boolean_switch(self):
        previous=ConditionalVelocity(width=8,blocks=1,modes=5)
        direct=ConditionalVelocity(width=8,blocks=1,modes=5,direct_conditioning=True)
        self.assertEqual(previous.lift[0].in_channels,35)
        self.assertEqual(direct.lift[0].in_channels,80)
        old_config={k:v for k,v in previous.config.items() if k!='direct_conditioning'}
        compatible=ConditionalVelocity(**old_config)
        compatible.load_state_dict(previous.state_dict(),strict=True)
        for value in ['true',1,None]:
            with self.assertRaises(ValueError):ConditionalVelocity(direct_conditioning=value)

    def test_same_seed_keeps_common_weights_identical_between_routes(self):
        torch.manual_seed(42)
        control=ConditionalVelocity(width=8,blocks=1,modes=5)
        torch.manual_seed(42)
        direct=ConditionalVelocity(width=8,blocks=1,modes=5,direct_conditioning=True)
        for key,value in control.state_dict().items():
            candidate=direct.state_dict()[key]
            if key=='lift.0.weight':candidate=candidate[:,:value.shape[1]]
            torch.testing.assert_close(candidate,value,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
