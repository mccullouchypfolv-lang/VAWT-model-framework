"""Periodic conditional residual velocity and differentiable explicit ODE solvers."""
import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F

from .model import reflect_height


class SpectralConv1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat)
                                   / math.sqrt(width))

    def forward(self, x):
        spectrum = torch.fft.rfft(x, dim=-1, norm='ortho')
        if self.modes > spectrum.shape[-1]:
            raise ValueError('spectral modes exceed angular grid resolution')
        output = spectrum.new_zeros(spectrum.shape)
        output[:, :, :self.modes] = torch.einsum(
            'bim,iom->bom', spectrum[:, :, :self.modes], self.weight)
        return torch.fft.irfft(output, n=x.shape[-1], dim=-1, norm='ortho')


def film_network(n_features, width):
    """Pointwise FiLM MLP with three 64-channel SiLU hidden layers."""
    return nn.Sequential(nn.Conv1d(n_features, 64, 1), nn.SiLU(),
                         nn.Conv1d(64, 64, 1), nn.SiLU(),
                         nn.Conv1d(64, 64, 1), nn.SiLU(),
                         nn.Conv1d(64, 2 * width, 1))


def apply_film(x, parameters):
    gamma, beta = parameters.chunk(2, dim=1)
    return (1 + gamma) * x + beta


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.register_buffer('frequencies', 2 * math.pi *
                             torch.exp(torch.linspace(0, math.log(1000), dimension // 2)))

    def forward(self, t):
        phase = t.reshape(-1, 1) * self.frequencies[None]
        return torch.cat((phase.sin(), phase.cos()), dim=1)


class VelocityBlock(nn.Module):
    def __init__(self, width, modes, modulation_channels):
        super().__init__()
        self.spectral = SpectralConv1d(width, modes)
        self.linear = nn.Conv1d(width, width, 1)
        self.film = film_network(modulation_channels, width)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x, modulation):
        features = F.silu(self.spectral(x) + self.linear(x))
        return self.norm(apply_film(features, self.film(modulation)))


class ConditionalVelocity(nn.Module):
    """Encode frozen LF conditions once and predict normalized residual velocity.

    ``prepare_condition`` returns angular features, global parameters and a
    cached encoded condition field. Keep its graph for training and recompute
    each training update; for sampling, compute once outside the ODE loop.
    """
    def __init__(self, width=64, blocks=4, modes=12, field_channels=32,
                 rotor_channels=32, height_width=8, n_conditions=2,
                 n_targets=3, lf_channels=3, lift_ratio=4, projection_ratio=4,
                 time_embedding_dim=32, architecture='floral-filmfno-v1',
                 direct_conditioning=False):
        super().__init__()
        self.config = dict(width=width, blocks=blocks, modes=modes,
                           field_channels=field_channels, rotor_channels=rotor_channels,
                           height_width=height_width, n_conditions=n_conditions,
                           n_targets=n_targets, lf_channels=lf_channels,
                           lift_ratio=lift_ratio, projection_ratio=projection_ratio,
                           time_embedding_dim=time_embedding_dim)
        if any(isinstance(value, bool) or not isinstance(value, numbers.Integral)
               or value < 1 for value in self.config.values()):
            raise ValueError('all model dimensions must be positive integers')
        if time_embedding_dim < 2 or time_embedding_dim % 2:
            raise ValueError('time_embedding_dim must be even and at least 2')
        if architecture != 'floral-filmfno-v1':
            raise ValueError('unsupported residual velocity architecture')
        if not isinstance(direct_conditioning,bool):
            raise ValueError('direct_conditioning must be a boolean')
        self.config['architecture'] = architecture
        self.config['direct_conditioning'] = direct_conditioning
        self.height_lift = nn.Conv2d(field_channels + 1, height_width, (3, 1))
        n_features = lf_channels + rotor_channels + height_width + 2
        self.condition_encoder = nn.Sequential(nn.Conv1d(n_features + n_conditions, width, 1),
                                               nn.SiLU(), nn.Conv1d(width, width, 1))
        self.time_embedding = FourierTimeEmbedding(time_embedding_dim)
        lift_channels=n_targets+time_embedding_dim
        self.lift = nn.Sequential(nn.Conv1d(lift_channels, width * lift_ratio, 1),
                                  nn.SiLU(), nn.Conv1d(width * lift_ratio, width, 1))
        modulation_channels = width + time_embedding_dim
        self.initial_film = film_network(modulation_channels, width)
        self.linear_spectral = SpectralConv1d(n_targets, modes)
        self.linear_pointwise = nn.Conv1d(n_targets, n_targets, 1)
        self.linear_film = film_network(modulation_channels, n_targets)
        self.blocks = nn.ModuleList([VelocityBlock(width, modes, modulation_channels)
                                     for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv1d(width, width * projection_ratio, 1), nn.SiLU(),
                                   nn.Conv1d(width * projection_ratio, n_targets, 1))
        if direct_conditioning:
            # Expand only after initializing all common modules, so equal seeds
            # preserve their weights and the original state/time lift columns.
            previous=self.lift[0]
            expanded=nn.Conv1d(lift_channels+n_features,width*lift_ratio,1)
            with torch.no_grad():
                expanded.weight[:,:lift_channels].copy_(previous.weight)
                expanded.bias.copy_(previous.bias)
            self.lift[0]=expanded

    def prepare_condition(self, lf, rotor, field, a, z, weights, theta):
        if lf.ndim != 3 or lf.shape[-1] != self.config['lf_channels']:
            raise ValueError('lf must be [B,T,lf_channels]')
        batch, nt, _ = lf.shape
        if (rotor.shape != (batch, self.config['rotor_channels'], nt)
                or field.shape != (batch, self.config['field_channels'], 20, nt)
                or a.shape != (batch, self.config['n_conditions'])
                or z.shape != (batch, 20) or weights.shape != (batch, 20)
                or theta.shape != (batch, nt)):
            raise ValueError('condition/coordinate/weight shape mismatch')
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('height weights must be finite and positive')
        height_coords = z[:, None, :, None].expand(-1, 1, -1, nt)
        encoded = F.gelu(self.height_lift(reflect_height(
            torch.cat((field, height_coords), dim=1), 1)))
        normalized_weights = weights / weights.sum(dim=-1, keepdim=True)
        pooled = (encoded * normalized_weights[:, None, :, None]).sum(dim=2)
        features = torch.cat((lf.transpose(1, 2), rotor, pooled,
                              theta.sin()[:, None], theta.cos()[:, None]), dim=1)
        encoded = self.condition_encoder(torch.cat((features, a[:, :, None].expand(-1, -1, nt)), dim=1))
        return {'features': features, 'modulation': a, 'encoded_features': encoded}

    def forward_branches(self, g, t, condition):
        """Return independent affine-linear and nonlinear velocity terms."""
        if g.ndim != 3 or g.shape[-1] != self.config['n_targets']:
            raise ValueError('g must be [B,T,n_targets]')
        batch, nt, _ = g.shape
        if t.shape not in ((batch,), (batch, 1, 1)):
            raise ValueError('t must be [B] or [B,1,1]')
        features, a = condition['features'], condition['modulation']
        n_features = self.config['lf_channels'] + self.config['rotor_channels'] + self.config['height_width'] + 2
        if features.shape != (batch, n_features, nt) or a.shape != (batch, self.config['n_conditions']):
            raise ValueError('prepared condition shape mismatch')
        encoded = condition.get('encoded_features')
        if encoded is None:
            raise ValueError('encoded condition missing; call prepare_condition first')
        if encoded.shape != (batch, self.config['width'], nt):
            raise ValueError('encoded condition shape mismatch')
        time = self.time_embedding(t).unsqueeze(-1).expand(-1, -1, nt)
        modulation = torch.cat((encoded, time), dim=1)
        raw = g.transpose(1, 2)
        linear = apply_film(self.linear_spectral(raw) + self.linear_pointwise(raw),
                            self.linear_film(modulation))
        state_inputs=(raw,time,features) if self.config['direct_conditioning'] else (raw,time)
        x = apply_film(self.lift(torch.cat(state_inputs, dim=1)), self.initial_film(modulation))
        for block in self.blocks:
            x = block(x, modulation)
        return linear.transpose(1, 2), self.head(x).transpose(1, 2)

    def forward(self, g, t, condition):
        linear, nonlinear = self.forward_branches(g, t, condition)
        return linear + nonlinear


def integrate_velocity(model, source, condition, steps=16, method='heun',
                       atol=1e-5, rtol=1e-5, diagnostics=None):
    """Integrate v(g,t,condition) over [0,1]; preserve autograd if enabled.

    Euler uses ``steps`` evaluations; Heun uses ``2*steps`` evaluations.
    dopri5 selects steps adaptively using atol/rtol; ``steps`` does not set its
    step size. Tolerances apply to normalized residual state. Inference callers
    should choose their own ``torch.no_grad`` context.
    """
    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if method not in ('euler', 'heun', 'dopri5'):
        raise ValueError('method must be euler, heun or dopri5')
    if method == 'dopri5':
        if any(isinstance(v,bool) or not isinstance(v,numbers.Real) or not math.isfinite(v) or v<=0
               for v in (atol,rtol)):
            raise ValueError('atol and rtol must be finite positive tolerances')
        from torchdiffeq import odeint
        evaluations=0
        def velocity(time,state):
            nonlocal evaluations
            evaluations+=1
            time_batch=time.to(device=source.device,dtype=source.dtype).expand(source.shape[0])
            return model(state,time_batch,condition)
        result=odeint(velocity,source,source.new_tensor([0.,1.]),method='dopri5',
                      atol=atol,rtol=rtol,options={'max_num_steps':100000})[-1]
        if diagnostics is not None:diagnostics.update(method=method,nfe=evaluations,atol=atol,rtol=rtol)
        return result
    g = source
    dt = 1.0 / steps
    for index in range(steps):
        t = source.new_full((source.shape[0],), index / steps)
        velocity = model(g, t, condition)
        proposal = g + dt * velocity
        if method == 'heun':
            next_t = source.new_full((source.shape[0],), (index + 1) / steps)
            next_velocity = model(proposal, next_t, condition)
            g = g + .5 * dt * (velocity + next_velocity)
        else:
            g = proposal
    if diagnostics is not None:diagnostics.update(method=method,nfe=steps*(2 if method=='heun' else 1))
    return g
