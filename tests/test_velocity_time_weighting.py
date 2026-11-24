import unittest
import torch
from lf_operator.objectives import velocity_loss


class VelocityTimeWeightingTests(unittest.TestCase):
    def test_quadratic_weight_matches_per_sample_reference_and_gradient(self):
        prediction=torch.tensor([1.,2.,3.]).reshape(3,1,1).expand(3,4,2).clone().requires_grad_()
        target=torch.zeros_like(prediction);time=torch.tensor([0.,.5,1.]).reshape(3,1,1)
        actual=velocity_loss(prediction,target,time=time,time_weighting='floral_quadratic')
        expected=((1+2*time.reshape(3)**2)*(prediction-target).square().mean((1,2))).mean()
        torch.testing.assert_close(actual,expected)
        actual.backward()
        weights=1+2*time**2
        torch.testing.assert_close(prediction.grad,2*prediction.detach()*weights/prediction.numel())
        self.assertAlmostEqual(actual.item(),(1+1.5*4+3*9)/3,places=5)

    def test_batch_of_one_keeps_batch_dimension(self):
        p=torch.ones(1,8,3)
        self.assertEqual(float(velocity_loss(p,p*0,time=torch.ones(1,1,1),time_weighting='floral_quadratic')),3.)

    def test_uniform_mode_reproduces_original_mse(self):
        p=torch.randn(7,16,3);y=torch.randn_like(p)
        self.assertTrue(torch.equal(velocity_loss(p,y),((p-y)**2).mean()))
        self.assertTrue(torch.equal(velocity_loss(p,y,time=torch.rand(7),time_weighting='none'),velocity_loss(p,y)))

    def test_invalid_weighting_and_times_are_rejected(self):
        p=torch.ones(2,4,3)
        cases=[dict(time_weighting='unknown'),dict(time_weighting='floral_quadratic'),
               dict(time_weighting='floral_quadratic',time=torch.tensor([0.,1.1])),
               dict(time_weighting='floral_quadratic',time=torch.tensor([0.,float('nan')])),
               dict(time_weighting='floral_quadratic',time=torch.zeros(2,2))]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):velocity_loss(p,p,**kwargs)


if __name__=='__main__':unittest.main()
