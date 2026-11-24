import unittest
import torch
from torch.nn import functional as F
from lf_operator.objectives import velocity_loss


class HuberVelocityTests(unittest.TestCase):
    def test_time_weighting_and_bounded_outlier_gradient(self):
        p=torch.tensor([.5,2.,20.]).reshape(3,1,1).expand(3,4,3).clone().requires_grad_()
        t=torch.tensor([0.,.5,1.]);target=torch.zeros_like(p)
        loss=velocity_loss(p,target,time=t,time_weighting='floral_quadratic',regression='huber',huber_delta=1.)
        expected=((1+2*t.square())*F.huber_loss(p,target,reduction='none',delta=1.).mean((1,2))).mean()
        torch.testing.assert_close(loss,expected)
        loss.backward()
        torch.testing.assert_close(p.grad,p.detach().clamp(-1,1)*(1+2*t[:,None,None].square())/p.numel())

    def test_weighted_huber_and_invalid_options(self):
        p=torch.randn(2,8,3);target=torch.randn_like(p);weights=torch.tensor([1.,2.]);angular=torch.ones(8,3)*2
        loss=velocity_loss(p,target,regression='huber',huber_delta=.5,sample_weights=weights,angular_weights=angular)
        expected=(F.huber_loss(p,target,reduction='none',delta=.5).mul(angular).mean((1,2))*weights).mean()
        torch.testing.assert_close(loss,expected)
        torch.testing.assert_close(velocity_loss(p,target),((p-target)**2).mean(),rtol=0,atol=0)
        for delta in (0.,-1.,float('nan'),float('inf'),True):
            with self.subTest(delta=delta),self.assertRaises(ValueError):
                velocity_loss(p,target,regression='huber',huber_delta=delta)
        with self.assertRaises(ValueError):velocity_loss(p,target,regression='unknown')


if __name__=='__main__':unittest.main()
