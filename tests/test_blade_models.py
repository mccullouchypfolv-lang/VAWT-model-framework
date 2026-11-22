"""Parameter-only blade dimensions and checkpoint reconstruction."""
import io
import unittest

import torch

from lf_operator.model import FiLMFNO
from lf_operator.residual_flow import ConditionalVelocity


class BladeModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def fno_inputs(self):
        return (torch.randn(2, 20, 2, 32), torch.randn(2, 2),
                torch.linspace(0, 1, 20).expand(2, -1),
                torch.arange(32).float().expand(2, -1) * (2 * torch.pi / 32),
                torch.ones(2, 20))

    def velocity_inputs(self, lf_channels):
        _, a, z, theta, weights = self.fno_inputs()
        return [torch.randn(2, 32, lf_channels), torch.randn(2, 8, 32),
                torch.randn(2, 8, 20, 32), a, z, weights, theta]

    def roundtrip(self, model):
        stream = io.BytesIO()
        torch.save({'config': model.config, 'state_dict': model.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        clone = type(model)(**checkpoint['config'])
        clone.load_state_dict(checkpoint['state_dict'], strict=True)
        return clone

    def test_three_target_fno_gradients_and_checkpoint(self):
        model = FiLMFNO(width=8, blocks=1,
                        modes_z=3, modes_theta=5, pad_z=4)
        inputs = self.fno_inputs()
        prediction = model(*inputs)
        self.assertEqual(prediction.shape, (2, 32, 3))
        prediction.square().mean().backward()
        self.assertTrue(torch.all(model.head[-1].weight.grad.abs().sum((1, 2)) > 0))
        torch.testing.assert_close(self.roundtrip(model)(*inputs), prediction)

    def test_three_target_velocity_condition_gradients_and_checkpoint(self):
        model = ConditionalVelocity(width=8, blocks=1,
                                    modes=5, field_channels=8, rotor_channels=8)
        inputs = [value.requires_grad_() for value in self.velocity_inputs(3)]
        condition = model.prepare_condition(*inputs)
        g = torch.randn(2, 32, 3, requires_grad=True)
        t = torch.tensor([.2, .7], requires_grad=True)
        velocity = model(g, t, condition)
        self.assertEqual(velocity.shape, g.shape)
        velocity.square().mean().backward()
        for value in inputs + [g, t]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        torch.testing.assert_close(self.roundtrip(model)(g, t, condition), velocity)
        with self.assertRaises(ValueError):
            model.prepare_condition(*self.velocity_inputs(2))
        with self.assertRaises(ValueError):
            model(g[..., :1], t, condition)

    def test_parameter_only_default_dimensions(self):
        fno = FiLMFNO(width=8, blocks=1, modes_z=3, modes_theta=5, pad_z=4)
        self.assertEqual((fno.config['n_features'], fno.config['n_conditions'],
                          fno.config['n_outputs']), (2, 2, 3))
        velocity = ConditionalVelocity(width=8, blocks=1, modes=5,
                                       field_channels=8, rotor_channels=8)
        self.assertEqual((velocity.config['n_conditions'], velocity.config['n_targets'],
                          velocity.config['lf_channels']), (2, 3, 3))

    def test_new_channel_dimensions_require_positive_integers(self):
        for value in [0, -1, 1.5, True]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FiLMFNO(n_outputs=value)
                with self.assertRaises(ValueError):
                    ConditionalVelocity(n_targets=value)
                with self.assertRaises(ValueError):
                    ConditionalVelocity(lf_channels=value)


if __name__ == '__main__':
    unittest.main()
