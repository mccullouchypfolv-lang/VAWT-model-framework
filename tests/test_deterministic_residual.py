import unittest
import torch

from lf_operator.deterministic_residual import ConditionalResidualMean


class DeterministicResidualTests(unittest.TestCase):
    def test_condition_only_prediction_and_resolution_transfer(self):
        torch.manual_seed(42)
        model=ConditionalResidualMean(width=8,blocks=2,modes=6,height_width=4,
            direct_conditioning=True,lift_ratio=2,projection_ratio=2)
        for n in [16,32]:
            condition=model.prepare_condition(torch.randn(2,n,3),torch.randn(2,32,n),
                torch.randn(2,32,20,n),torch.randn(2,2),torch.linspace(0,1,20).repeat(2,1),
                torch.ones(2,20),torch.arange(n).float()[None].repeat(2,1)*2*torch.pi/n)
            out=model(condition)
            self.assertEqual(out.shape,(2,n,3))
            self.assertTrue(torch.isfinite(out).all())
            torch.testing.assert_close(out,model(condition),rtol=0,atol=0)
            out.square().mean().backward()
        self.assertIsNotNone(model.head[-1].weight.grad)
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()),0)

    def test_supervised_residual_loss_equals_corrected_curve_loss(self):
        from lf_operator.blade_pipeline import encode_residual,decode_residual
        import numpy as np
        rng=np.random.default_rng(3);residual=rng.normal(size=(2,16,3));prediction=rng.normal(size=residual.shape)
        stats=dict(mean_profile=rng.normal(size=(16,3)).tolist(),scale=[2.,.2,.02])
        loss=np.mean((prediction-encode_residual(residual,stats))**2)
        corrected=decode_residual(prediction,stats)
        physical=np.mean(((corrected-residual)/np.array(stats['scale']))**2)
        np.testing.assert_allclose(loss,physical,rtol=1e-6)


if __name__=='__main__':unittest.main()
