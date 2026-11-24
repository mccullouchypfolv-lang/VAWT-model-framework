import unittest
import torch
from lf_operator.objectives import complex_spectral_loss


class ComplexSpectralTests(unittest.TestCase):
    def test_real_fft_loss_matches_full_complex_fft_with_time_weighting(self):
        for n in (15,16):
            torch.manual_seed(42)
            p=torch.randn(3,n,3,dtype=torch.float64,requires_grad=True);target=torch.randn_like(p)
            t=torch.tensor([0.,.5,1.],dtype=torch.float64)
            actual=complex_spectral_loss(p,target,time=t,time_weighting='floral_quadratic')
            expected=((1+2*t*t)*torch.fft.fft(p-target,dim=1,norm='ortho').abs().mean((1,2))).mean()
            torch.testing.assert_close(actual,expected)
            a=torch.autograd.grad(actual,p,retain_graph=True)[0];b=torch.autograd.grad(expected,p)[0]
            torch.testing.assert_close(a,b)

    def test_phase_sensitive_and_distinct_from_squared_error(self):
        t=torch.arange(32,dtype=torch.float64)*2*torch.pi/32
        p=torch.cos(t)[None,:,None].requires_grad_();target=torch.sin(t)[None,:,None]
        a=complex_spectral_loss(p,target);b=complex_spectral_loss(2*p,2*target)
        self.assertGreater(float(a),0)
        torch.testing.assert_close(b,2*a)  # L1 homogeneity, not MSE's factor four
        a.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_zero_error_has_finite_zero_gradient(self):
        p=torch.zeros(2,16,3,requires_grad=True)
        loss=complex_spectral_loss(p,torch.zeros_like(p));loss.backward()
        self.assertEqual(float(loss),0)
        self.assertEqual(float(p.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
