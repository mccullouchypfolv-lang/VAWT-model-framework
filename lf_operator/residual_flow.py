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
    def __init__(self, dimension, max_frequency=1000.):
        super().__init__()
        if isinstance(max_frequency,bool) or not isinstance(max_frequency,numbers.Real) or not math.isfinite(max_frequency) or max_frequency<1:
            raise ValueError('maximum FM time frequency must be finite and >=1')
        self.register_buffer('frequencies', 2 * math.pi *
                             torch.exp(torch.linspace(0, math.log(max_frequency), dimension // 2)))

    def forward(self, t):
        phase = t.reshape(-1, 1) * self.frequencies[None]
        return torch.cat((phase.sin(), phase.cos()), dim=1)


class FloralTimeEmbedding(nn.Module):
    """Official five fixed frequencies, producing ten sine/cosine channels."""
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies',2*math.pi*torch.logspace(-1,2,5))

    def forward(self,t):
        phase=t.reshape(-1,1)*self.frequencies[None]
        return torch.cat((phase.sin(),phase.cos()),dim=1)


def floral_grid(n,*,device,dtype):
    """neuraloperator 1.0.2 GridEmbeddingND uses linspace(0,1,n+1)[:-1]."""
    return torch.linspace(0.,1.,n+1,device=device,dtype=dtype)[:-1]


class FloralVelocityBlock(nn.Module):
    """Official postactivation ordering, no norm, channel MLP and soft-gate skip.

    Local SpectralConv1d keeps the project's rFFT-coefficient modes convention.
    FNOBlocks reference: neuraloperator 1.0.2, forward_with_postactivation.
    """
    def __init__(self,width,modes,modulation_channels,last=False):
        super().__init__()
        self.spectral=SpectralConv1d(width,modes)
        self.spectral_bias=nn.Parameter(torch.zeros(1,width,1))
        self.linear=nn.Conv1d(width,width,1,bias=False)
        self.channel_mlp=nn.Sequential(nn.Conv1d(width,max(1,width//2),1),nn.GELU(),
                                       nn.Conv1d(max(1,width//2),width,1))
        self.channel_gate=nn.Parameter(torch.ones(1,width,1))
        self.film=film_network(modulation_channels,width)
        self.last=last

    def forward(self,x,modulation):
        skip=self.channel_gate*x
        x=F.gelu(self.spectral(x)+self.spectral_bias+self.linear(x))
        x=self.channel_mlp(x)+skip
        if not self.last:x=F.gelu(x)
        return apply_film(x,self.film(modulation))


class VelocityBlock(nn.Module):
    def __init__(self, width, modes, modulation_channels):
        super().__init__()
        self.spectral = SpectralConv1d(width, modes)
        self.linear = nn.Conv1d(width, width, 1)
        self.film = film_network(modulation_channels, width)
        self.norm = nn.GroupNorm(1, width)

    def add_channel_mlp(self):
        """Opt-in pair only; called after every shared module is initialized."""
        if hasattr(self,'channel_mlp'):raise ValueError('channel MLP already enabled')
        width=self.linear.in_channels
        self.channel_mlp=nn.Sequential(nn.Conv1d(width,max(1,width//2),1),nn.GELU(),
                                       nn.Conv1d(max(1,width//2),width,1))
        self.channel_gate=nn.Parameter(torch.ones(1,width,1))

    def forward(self, x, modulation):
        features = F.silu(self.spectral(x) + self.linear(x))
        if hasattr(self,'channel_mlp'):
            features=self.channel_mlp(features)+self.channel_gate*x
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
                 direct_conditioning=False,explicit_physical_conditioning=False,geometry_width=8,channel_mlp_skip=False,
                 time_max_frequency=1000.,scalar_geometry_film=False):
        super().__init__()
        self.config = dict(width=width, blocks=blocks, modes=modes,
                           field_channels=field_channels, rotor_channels=rotor_channels,
                           height_width=height_width, n_conditions=n_conditions,
                           n_targets=n_targets, lf_channels=lf_channels,
                           lift_ratio=lift_ratio, projection_ratio=projection_ratio,
                           time_embedding_dim=time_embedding_dim,geometry_width=geometry_width)
        if any(isinstance(value, bool) or not isinstance(value, numbers.Integral)
               or value < 1 for value in self.config.values()):
            raise ValueError('all model dimensions must be positive integers')
        if time_embedding_dim < 2 or time_embedding_dim % 2:
            raise ValueError('time_embedding_dim must be even and at least 2')
        if architecture not in ('floral-filmfno-v1','floral-filmfno-aligned-v2'):
            raise ValueError('unsupported residual velocity architecture')
        aligned=architecture=='floral-filmfno-aligned-v2'
        if not isinstance(scalar_geometry_film,bool) or (scalar_geometry_film and (aligned or explicit_physical_conditioning)):
            raise ValueError('scalar geometry FiLM is an isolated option for the original architecture')
        self.config['scalar_geometry_film']=scalar_geometry_film
        if aligned and time_max_frequency!=1000.:
            raise ValueError('aligned model uses its own fixed official time frequencies')
        self.config['time_max_frequency']=time_max_frequency
        if not isinstance(channel_mlp_skip,bool) or (aligned and channel_mlp_skip):
            raise ValueError('channel_mlp_skip must be boolean and is only an ablation of the original architecture')
        if aligned and (time_embedding_dim!=10 or explicit_physical_conditioning):
            raise ValueError('aligned architecture requires time_embedding_dim=10 and existing LF/height conditions')
        if not isinstance(direct_conditioning,bool):
            raise ValueError('direct_conditioning must be a boolean')
        self.config['architecture'] = architecture
        self.config['channel_mlp_skip']=channel_mlp_skip
        self.config['direct_conditioning'] = direct_conditioning
        if not isinstance(explicit_physical_conditioning,bool):
            raise ValueError('explicit_physical_conditioning must be a boolean')
        if explicit_physical_conditioning and not direct_conditioning:
            raise ValueError('explicit physical conditioning requires direct_conditioning')
        self.config['explicit_physical_conditioning']=explicit_physical_conditioning
        self.height_lift = nn.Conv2d(field_channels + 1, height_width, (3, 1))
        n_features = lf_channels + rotor_channels + height_width + (1 if aligned else 2)
        condition_hidden=width*lift_ratio if aligned else width
        self.condition_encoder = nn.Sequential(nn.Conv1d(n_features + n_conditions, condition_hidden, 1),
                                               nn.GELU() if aligned else nn.SiLU(), nn.Conv1d(condition_hidden, width, 1))
        self.time_embedding = FloralTimeEmbedding() if aligned else FourierTimeEmbedding(time_embedding_dim,max_frequency=time_max_frequency)
        lift_channels=n_targets+time_embedding_dim+(1 if aligned and not direct_conditioning else 0)
        self.lift = nn.Sequential(nn.Conv1d(lift_channels, width * lift_ratio, 1),
                                  nn.GELU() if aligned else nn.SiLU(), nn.Conv1d(width * lift_ratio, width, 1))
        modulation_channels = width if aligned else width + time_embedding_dim
        self.initial_film = film_network(modulation_channels, width)
        if aligned:
            self.linear_operator=nn.Sequential(nn.Conv1d(n_targets+time_embedding_dim,width,1),
                nn.Conv1d(width,width,1),nn.Conv1d(width,width,1),nn.Conv1d(width,n_targets,1))
            self.linear_film=film_network(n_features+n_conditions-1,n_targets)
            self.blocks=nn.ModuleList([FloralVelocityBlock(width,modes,modulation_channels,last=i==blocks-1) for i in range(blocks)])
        else:
            self.linear_spectral = SpectralConv1d(n_targets, modes)
            self.linear_pointwise = nn.Conv1d(n_targets, n_targets, 1)
            self.linear_film = film_network(modulation_channels, n_targets)
            self.blocks = nn.ModuleList([VelocityBlock(width, modes, modulation_channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv1d(width, width * projection_ratio, 1), nn.GELU() if aligned else nn.SiLU(),
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

        if explicit_physical_conditioning:
            # Initialize additions last, preserving all v1 common parameters.
            self.geometry_encoder=nn.Sequential(nn.Conv1d(3,geometry_width,3),nn.SiLU(),
                                                nn.Conv1d(geometry_width,geometry_width,1),nn.SiLU())
            previous=self.condition_encoder[0]
            expanded=nn.Conv1d(n_features+n_conditions+geometry_width,width,1)
            with torch.no_grad():
                expanded.weight[:,:previous.in_channels].copy_(previous.weight)
                expanded.bias.copy_(previous.bias)
            self.condition_encoder[0]=expanded
            previous=self.lift[0]
            expanded=nn.Conv1d(previous.in_channels+n_conditions+geometry_width,width*lift_ratio,1)
            with torch.no_grad():
                expanded.weight[:,:previous.in_channels].copy_(previous.weight)
                expanded.bias.copy_(previous.bias)
            self.lift[0]=expanded

        if channel_mlp_skip:
            # Preserve same-seed initialization of ALL existing parameters.
            for block in self.blocks:block.add_channel_mlp()

        if scalar_geometry_film:
            # Append c/tc only to the condition lifting. Existing U/TSR columns,
            # all main lift columns and all shared initialization stay intact.
            previous=self.condition_encoder[0]
            expanded=nn.Conv1d(previous.in_channels+2,previous.out_channels,1)
            with torch.no_grad():
                expanded.weight[:,:previous.in_channels].copy_(previous.weight)
                expanded.bias.copy_(previous.bias)
            self.condition_encoder[0]=expanded

    def prepare_condition(self, lf, rotor, field, a, z, weights, theta,geometry=None):
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
        aligned=self.config['architecture']=='floral-filmfno-aligned-v2'
        coords=(floral_grid(nt,device=lf.device,dtype=lf.dtype)[None,None].expand(batch,-1,-1),) if aligned else (theta.sin()[:,None],theta.cos()[:,None])
        features = torch.cat((lf.transpose(1, 2), rotor, pooled,*coords), dim=1)
        operating=a[:,:,None].expand(-1,-1,nt)
        if self.config['explicit_physical_conditioning']:
            if geometry is None or geometry.shape!=(batch,20,2) or not torch.isfinite(geometry).all():
                raise ValueError('normalized geometry must be finite [B,20,2]')
            profile=torch.cat((geometry.transpose(1,2),z[:,None,:]),dim=1)
            geometric=self.geometry_encoder(F.pad(profile,(1,1),mode='reflect'))
            geometric=(geometric*normalized_weights[:,None,:]).sum(-1)
            features=torch.cat((features,operating,geometric[:,:,None].expand(-1,-1,nt)),dim=1)
            encoded=self.condition_encoder(features)
        else:
            condition_inputs=[features,operating]
            if self.config['scalar_geometry_film']:
                if geometry is None or geometry.shape!=(batch,20,2) or not torch.isfinite(geometry).all():
                    raise ValueError('normalized chord/thickness geometry required')
                if not torch.allclose(geometry,geometry[:,:1].expand_as(geometry),rtol=1e-6,atol=1e-6):
                    raise ValueError('scalar geometry interface requires spanwise-uniform chord/thickness')
                condition_inputs.append(geometry[:,0,:,None].expand(-1,-1,nt))
            encoded=self.condition_encoder(torch.cat(condition_inputs,dim=1))
        result={'features': features, 'modulation': a, 'encoded_features': encoded}
        # Official linear FiLM uses raw conditions, before positional embedding.
        # The grid remains in state/condition lifting, not in raw linear FiLM.
        if aligned:result['raw_condition']=torch.cat((features[:,:-1],operating),dim=1)
        return result

    def forward_branches(self, g, t, condition):
        """Return independent affine-linear and nonlinear velocity terms."""
        if g.ndim != 3 or g.shape[-1] != self.config['n_targets']:
            raise ValueError('g must be [B,T,n_targets]')
        batch, nt, _ = g.shape
        if t.shape not in ((batch,), (batch, 1, 1)):
            raise ValueError('t must be [B] or [B,1,1]')
        features, a = condition['features'], condition['modulation']
        aligned=self.config['architecture']=='floral-filmfno-aligned-v2'
        n_features = self.config['lf_channels'] + self.config['rotor_channels'] + self.config['height_width'] + (1 if aligned else 2)
        if self.config['explicit_physical_conditioning']:
            n_features+=self.config['n_conditions']+self.config['geometry_width']
        if features.shape != (batch, n_features, nt) or a.shape != (batch, self.config['n_conditions']):
            raise ValueError('prepared condition shape mismatch')
        encoded = condition.get('encoded_features')
        if encoded is None:
            raise ValueError('encoded condition missing; call prepare_condition first')
        if encoded.shape != (batch, self.config['width'], nt):
            raise ValueError('encoded condition shape mismatch')
        time = self.time_embedding(t).unsqueeze(-1).expand(-1, -1, nt)
        modulation = encoded if aligned else torch.cat((encoded, time), dim=1)
        raw = g.transpose(1, 2)
        if aligned:
            linear=apply_film(self.linear_operator(torch.cat((raw,time),dim=1)),self.linear_film(condition['raw_condition']))
        else:
            linear = apply_film(self.linear_spectral(raw) + self.linear_pointwise(raw),self.linear_film(modulation))
        state_inputs=(raw,time,features) if self.config['direct_conditioning'] else ((raw,time,features[:,-1:]) if aligned else (raw,time))
        x = apply_film(self.lift_state(state_inputs,condition), self.initial_film(modulation))
        for index in range(len(self.blocks)):
            x = self.nonlinear_block(index,x,modulation,condition)
        return linear.transpose(1, 2), self.head(x).transpose(1, 2)

    def nonlinear_block(self,index,x,modulation,condition):
        return self.blocks[index](x,modulation)

    def lift_state(self,state_inputs,condition):
        return self.lift(torch.cat(state_inputs,dim=1))

    def forward(self, g, t, condition):
        linear, nonlinear = self.forward_branches(g, t, condition)
        return linear + nonlinear


def create_velocity_model(**config):
    if config.get('architecture')=='angular-state-film-v1':
        from .angular_state_velocity import AngularStateVelocity
        return AngularStateVelocity(**config)
    if config.get('architecture')=='layer-wind-film-path-v1':
        from .layer_wind_velocity import LayerWindVelocity
        return LayerWindVelocity(**config)
    if config.get('architecture')=='physical-film-path-v1':
        from .physical_film_velocity import PhysicalFiLMVelocity
        return PhysicalFiLMVelocity(**config)
    if config.get('architecture')=='rotor-only-filmfno-v1':
        from .rotor_only_velocity import RotorOnlyVelocity
        return RotorOnlyVelocity(**config)
    """Load opt-in physical-condition models without changing old checkpoints."""
    if config.get('architecture')=='target-decoupled-film-v1':
        from .target_decoupled_flow import TargetDecoupledVelocity
        return TargetDecoupledVelocity(**config)
    if config.get('architecture')=='raw-physical-film-v1':
        from .raw_condition_velocity import RawConditionVelocity
        return RawConditionVelocity(**config)
    return ConditionalVelocity(**config)


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
