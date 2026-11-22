"""Conditional residual velocity and explicit ODE behavior."""
import json
import math
import unittest

import torch

from lf_operator.residual_flow import ConditionalVelocity, integrate_velocity


class ResidualFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        torch.set_num_threads(1)
        self.model = ConditionalVelocity(width=8, blocks=2, modes=5)
        b, nt = 2, 32
        self.inputs = [torch.randn(b, nt, 3), torch.randn(b, 32, nt),
                       torch.randn(b, 32, 20, nt), torch.randn(b, 2),
                       torch.linspace(0, 1, 20).expand(b, -1).clone(),
                       torch.rand(b, 20) + .1,
                       torch.arange(nt).expand(b, -1).float() * (2 * math.pi / nt)]

    def test_shape_config_and_gradients_through_every_condition(self):
        inputs = [value.requires_grad_() for value in self.inputs]
        condition = self.model.prepare_condition(*inputs)
        self.assertEqual(condition['features'].shape, (2, 45, 32))
        g = torch.randn(2, 32, 3, requires_grad=True)
        t = torch.tensor([.2, .7], requires_grad=True)
        velocity = self.model(g, t, condition)
        self.assertEqual(velocity.shape, g.shape)
        velocity.square().mean().backward()
        for value in inputs + [g, t]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        recreated = ConditionalVelocity(**json.loads(json.dumps(self.model.config)))
        recreated.load_state_dict(self.model.state_dict())
        self.assertTrue(torch.allclose(velocity, recreated(g, t[:, None, None], condition)))

    def test_circular_grid_permutation_equivariance(self):
        g = torch.randn(2, 32, 3)
        condition = self.model.prepare_condition(*self.inputs)
        expected = self.model(g, torch.tensor([.2, .7]), condition).roll(7, dims=1)
        lf, rotor, field, a, z, weights, theta = self.inputs
        rotated = self.model.prepare_condition(lf.roll(7, 1), rotor.roll(7, 2),
                    field.roll(7, 3), a, z, weights, theta.roll(7, 1))
        actual = self.model(g.roll(7, 1), torch.tensor([.2, .7]), rotated)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_physical_weights_scale_invariance_and_validation(self):
        condition = self.model.prepare_condition(*self.inputs)
        values = list(self.inputs)
        values[5] = values[5] * 7
        actual = self.model.prepare_condition(*values)
        torch.testing.assert_close(actual['features'], condition['features'])
        values[5] = -values[5]
        with self.assertRaises(ValueError):
            self.model.prepare_condition(*values)

    def test_independent_branches_sum_and_linear_branch_is_affine(self):
        self.assertTrue(hasattr(self.model, 'forward_branches'))
        condition = self.model.prepare_condition(*self.inputs)
        g1, g2 = torch.randn(2, 32, 3), torch.randn(2, 32, 3)
        t = torch.tensor([.2, .7])
        linear, nonlinear = self.model.forward_branches(g1, t, condition)
        torch.testing.assert_close(self.model(g1, t, condition), linear + nonlinear)
        linear2, _ = self.model.forward_branches(g2, t, condition)
        mixed, _ = self.model.forward_branches(.3 * g1 + .7 * g2, t, condition)
        torch.testing.assert_close(mixed, .3 * linear + .7 * linear2)
        with torch.no_grad():
            for parameter in self.model.head.parameters():
                parameter.zero_()
        after_linear, after_nonlinear = self.model.forward_branches(g1, t, condition)
        torch.testing.assert_close(after_linear, linear)
        torch.testing.assert_close(after_nonlinear, torch.zeros_like(nonlinear))
        torch.testing.assert_close(self.model(g1, t, condition), linear)

    def test_lf_controls_spatial_film_in_both_branches(self):
        self.assertTrue(hasattr(self.model, 'forward_branches'))
        g, t = torch.randn(2, 32, 3), torch.tensor([.2, .7])
        condition = self.model.prepare_condition(*self.inputs)
        changed = list(self.inputs)
        changed[0] = changed[0] + 3
        other = self.model.prepare_condition(*changed)
        for first, second in zip(self.model.forward_branches(g, t, condition),
                                 self.model.forward_branches(g, t, other)):
            self.assertGreater((first - second).abs().max().item(), 1e-5)
        captured = []
        handle = self.model.blocks[0].film.register_forward_hook(
            lambda module, args, output: captured.append(output.detach()))
        self.model(g, t, condition)
        handle.remove()
        self.assertEqual(captured[0].shape, (2, 16, 32))
        self.assertGreater(captured[0].var(dim=-1).sum().item(), 0)

    def test_fourier_time_embedding_normalization_and_default_architecture(self):
        default = ConditionalVelocity()
        self.assertEqual(default.config.get('architecture'), 'floral-filmfno-v1')
        self.assertEqual(default.config['width'], 64)
        self.assertEqual(default.config['blocks'], 4)
        self.assertEqual(default.config['time_embedding_dim'], 32)
        embedding = self.model.time_embedding(torch.tensor([0., .25]))
        self.assertEqual(embedding.shape, (2, 32))
        torch.testing.assert_close(embedding[:, :16].square() +
                                   embedding[:, 16:].square(), torch.ones(2, 16))
        self.assertFalse(torch.allclose(embedding[0], embedding[1]))
        self.assertEqual(self.model.lift[0].in_channels, 35)
        self.assertEqual(self.model.lift[0].out_channels, 32)
        self.assertEqual(self.model.head[0].out_channels, 32)
        for block in self.model.blocks:
            self.assertIsInstance(block.norm, torch.nn.GroupNorm)
            self.assertEqual(block.norm.num_groups, 1)
            hidden = [layer for layer in block.film if isinstance(layer, torch.nn.Conv1d)][:-1]
            self.assertEqual([layer.out_channels for layer in hidden], [64, 64, 64])

    def test_architecture_and_dimension_validation(self):
        self.assertIn('architecture', self.model.config)
        with self.assertRaises(ValueError):
            ConditionalVelocity(architecture='legacy')
        for key in ['width', 'blocks', 'modes', 'lift_ratio', 'projection_ratio',
                    'time_embedding_dim']:
            for invalid in [0, -1, True, 1.5]:
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    ConditionalVelocity(**{key: invalid})
        with self.assertRaises(ValueError):
            ConditionalVelocity(time_embedding_dim=3)

    def test_missing_cached_condition_is_rejected(self):
        condition = self.model.prepare_condition(*self.inputs)
        del condition['encoded_features']
        with self.assertRaisesRegex(ValueError, 'prepare_condition'):
            self.model(torch.randn(2,32,3),torch.tensor([.2,.7]),condition)

    def test_constant_ode_and_evaluation_counts(self):
        source = torch.randn(2, 32, 3, requires_grad=True)
        for method, multiplier in [('euler', 1), ('heun', 2)]:
            times = []
            def constant(g, t, condition):
                times.append(t.detach().clone())
                self.assertIs(condition, self)
                return torch.ones_like(g) * 2
            actual = integrate_velocity(constant, source, self, steps=7, method=method)
            torch.testing.assert_close(actual, source + 2)
            self.assertEqual(len(times), multiplier * 7)
            self.assertEqual(times[0].shape, (2,))
            self.assertAlmostEqual(times[0][0].item(), 0.)
            if method == 'heun':
                self.assertAlmostEqual(times[-1][0].item(), 1.)
            actual.sum().backward()
            torch.testing.assert_close(source.grad, torch.ones_like(source))
            source.grad = None

    def test_linear_ode_convergence_orders(self):
        source = torch.ones(1, 8, 3, dtype=torch.float64)
        errors = {}
        for method in ['euler', 'heun']:
            errors[method] = []
            for steps in [8, 16]:
                actual = integrate_velocity(lambda g, t, c: g, source, None,
                                            steps=steps, method=method)
                errors[method].append((actual - math.e).abs().max().item())
        self.assertGreater(errors['euler'][0] / errors['euler'][1], 1.8)
        self.assertGreater(errors['heun'][0] / errors['heun'][1], 3.6)
        self.assertLess(errors['heun'][1], errors['euler'][1])

    def test_invalid_solver_settings(self):
        for steps in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                integrate_velocity(lambda g, t, c: g, torch.ones(1, 2, 1), None, steps=steps)
        with self.assertRaises(ValueError):
            integrate_velocity(lambda g, t, c: g, torch.ones(1, 2, 1), None, method='rk4')

    def test_dopri5_accuracy_gradients_and_adaptive_evaluation(self):
        source=torch.ones(2,8,3,dtype=torch.float64,requires_grad=True)
        diagnostics={}
        def velocity(g,t,condition):
            self.assertEqual(t.shape,(2,))
            self.assertIs(condition,self)
            return g
        result=integrate_velocity(velocity,source,self,method='dopri5',atol=1e-5,rtol=1e-5,diagnostics=diagnostics)
        torch.testing.assert_close(result,source*math.e,atol=1e-4,rtol=1e-4)
        result.sum().backward()
        torch.testing.assert_close(source.grad,torch.full_like(source,math.e),atol=1e-4,rtol=1e-4)
        self.assertGreater(diagnostics['nfe'],0)
        for tolerance in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                integrate_velocity(velocity,source,self,method='dopri5',atol=tolerance)


if __name__ == '__main__':
    unittest.main()
