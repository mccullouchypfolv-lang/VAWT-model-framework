import io
import unittest

import torch

from lf_operator.model import FiLMFNO, SpectralConv2d


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def inputs(self, batch=2, nt=32):
        return (torch.randn(batch,20,2,nt), torch.randn(batch,2),
                torch.linspace(.025,.975,20).expand(batch,-1),
                ((torch.arange(nt)+.5)*2*torch.pi/nt).expand(batch,-1),
                torch.ones(batch,20))

    def test_shapes_features_gradients_and_roundtrip(self):
        model = FiLMFNO(width=8, blocks=2, modes_z=4, modes_theta=6, pad_z=4)
        args = self.inputs()
        result = model(*args, return_features=True)
        self.assertEqual(result['prediction'].shape, (2,32,3))
        self.assertEqual(result['field_features'].shape, (2,8,20,32))
        self.assertEqual(result['rotor_features'].shape, (2,8,32))
        result['prediction'].square().mean().backward()
        self.assertGreater(model.blocks[0].film[-1].weight.grad.abs().sum().item(),0)
        model.eval()
        all_at_once = model(*args)
        single = model(*(a[:1] for a in args))
        torch.testing.assert_close(all_at_once[:1], single, atol=1e-6, rtol=1e-5)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        clone = FiLMFNO(**model.config).eval()
        clone.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(clone(*args), all_at_once)

    def test_height_aggregation_feeds_prediction_head_directly(self):
        model = FiLMFNO(width=8,blocks=1,modes_z=3,modes_theta=5,pad_z=4).eval()
        x,a,z,theta,w = self.inputs()
        w = torch.arange(1,21,dtype=x.dtype).expand(x.shape[0],-1)
        result = model(x,a,z,theta,w,return_features=True)
        expected = (result['field_features']*(w/w.sum(-1,keepdim=True))[:,None,:,None]).sum(2)
        torch.testing.assert_close(result['rotor_features'],expected)
        torch.testing.assert_close(result['prediction'],model.head(expected).transpose(1,2))
        self.assertFalse(any('phase_fusion' in key for key in model.state_dict()))

    def test_both_signed_height_modes_receive_gradients(self):
        layer = SpectralConv2d(4,3,5)
        result = layer(torch.randn(2,4,28,32))
        result.square().mean().backward()
        for w in (layer.weight_pos, layer.weight_neg):
            self.assertTrue(torch.isfinite(w.grad).all())
            self.assertGreater(w.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):
            SpectralConv2d(4,15,5)(torch.randn(1,4,20,32))

    def test_periodic_roll_equivariance(self):
        model = FiLMFNO(width=8,blocks=1,modes_z=3,modes_theta=5,pad_z=4).eval()
        x,a,z,theta,w = self.inputs()
        y = model(x,a,z,theta,w)
        shifted = model(x.roll(3,-1),a,z,theta.roll(3,-1),w)
        torch.testing.assert_close(shifted,y.roll(3,1),atol=2e-6,rtol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_backward_with_strict_determinism(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            model = FiLMFNO(width=8,blocks=1,modes_z=3,modes_theta=5,pad_z=4).cuda()
            args = tuple(v.cuda() for v in self.inputs())
            prediction = model(*args)
            prediction.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == '__main__':
    unittest.main()
