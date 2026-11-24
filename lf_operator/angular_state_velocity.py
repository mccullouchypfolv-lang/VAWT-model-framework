"""Azimuthal periodic memory states conditioning the default residual velocity.

The state coordinate is physical azimuth, never the artificial FM time. Latents
are learned memory features, not identified separation or vortex measurements.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .residual_flow import ConditionalVelocity


def periodic_relaxation(forcing,tau):
    """Periodic steady state of tau*ds/dtheta+s=forcing, on a uniform circle."""
    if forcing.ndim!=3 or tau.shape!=(forcing.shape[1],) or forcing.shape[-1]<3:
        raise ValueError('forcing [B,D,N>=3] and relaxation times [D] required')
    if not torch.isfinite(tau).all() or (tau<=0).any():
        raise ValueError('relaxation times must be finite and positive')
    n=forcing.shape[-1]
    k=torch.arange(n//2+1,device=forcing.device,dtype=forcing.dtype)
    transfer=1/(1+1j*tau[None,:,None]*k[None,None,:])
    return torch.fft.irfft(torch.fft.rfft(forcing,dim=-1,norm='ortho')*transfer,n=n,dim=-1,norm='ortho')


class AngularStateEncoder(nn.Module):
    def __init__(self,in_channels,channels=8,width=32):
        super().__init__()
        self.forcing=nn.Sequential(nn.Conv1d(in_channels,width,1),nn.SiLU(),nn.Conv1d(width,channels,1))
        initial_tau=torch.logspace(math.log10(.05),math.log10(2.),channels)
        self.raw_tau=nn.Parameter(torch.expm1(initial_tau).log())

    def forward(self,x):
        return periodic_relaxation(self.forcing(x),F.softplus(self.raw_tau)+1e-4)


class AngularStateVelocity(ConditionalVelocity):
    def __init__(self,state_channels=8,state_width=32,architecture='angular-state-film-v1',**options):
        if architecture!='angular-state-film-v1':raise ValueError('unsupported angular state architecture')
        for value in (state_channels,state_width):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('state dimensions must be positive integers')
        if not options.get('direct_conditioning',False) or any(options.get(key,False) for key in
                ('explicit_physical_conditioning','scalar_geometry_film','channel_mlp_skip')):
            raise ValueError('angular state trial requires the default direct-conditioned backbone')
        super().__init__(architecture='floral-filmfno-v1',**options)
        # Initialize additions after all shared modules, preserving common seeds.
        self.state_encoder=AngularStateEncoder(2*self.config['lf_channels']+self.config['n_conditions']+4,state_channels,state_width)
        self.state_to_condition=nn.Conv1d(state_channels,self.config['width'],1)
        nn.init.zeros_(self.state_to_condition.weight);nn.init.zeros_(self.state_to_condition.bias)
        previous=self.lift[0]
        expanded=nn.Conv1d(previous.in_channels+state_channels,previous.out_channels,1)
        with torch.no_grad():
            expanded.weight[:,:previous.in_channels].copy_(previous.weight)
            expanded.weight[:,previous.in_channels:].zero_();expanded.bias.copy_(previous.bias)
        self.lift[0]=expanded
        self.config.update(architecture=architecture,state_channels=state_channels,state_width=state_width)

    def prepare_condition(self,lf,rotor,field,a,z,weights,theta,scalar_geometry=None):
        if scalar_geometry is None or scalar_geometry.shape!=(len(lf),2) or not torch.isfinite(scalar_geometry).all():
            raise ValueError('angular state encoder requires finite scalar chord/thickness [B,2]')
        n=lf.shape[1];delta=2*math.pi/n
        spacing=torch.remainder(theta.roll(-1,-1)-theta,2*math.pi)
        if not torch.allclose(spacing,spacing.new_full(spacing.shape,delta),rtol=1e-4,atol=1e-5):
            raise ValueError('angular memory requires a uniform periodic angular grid')
        condition=super().prepare_condition(lf,rotor,field,a,z,weights,theta)
        curve=lf.transpose(1,2)
        slope=(curve.roll(-1,-1)-curve.roll(1,-1))/(2*delta)
        physical=torch.cat((a,scalar_geometry),dim=1)[:,:,None].expand(-1,-1,n)
        state=self.state_encoder(torch.cat((curve,slope,physical,theta.sin()[:,None],theta.cos()[:,None]),dim=1))
        condition['angular_state']=state
        condition['encoded_features']=condition['encoded_features']+self.state_to_condition(state)
        return condition

    def lift_state(self,state_inputs,condition):
        return self.lift(torch.cat((*state_inputs,condition['angular_state']),dim=1))
