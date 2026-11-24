"""Shared spectral kernels with target-private FiLM streams and projections."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .residual_flow import ConditionalVelocity,apply_film


def _select_outputs(module,rows):
    result=deepcopy(module)
    result[-1].weight=nn.Parameter(module[-1].weight[rows].clone())
    result[-1].bias=nn.Parameter(module[-1].bias[rows].clone())
    result[-1].out_channels=len(rows)
    return result


class TargetModulatedBlock(nn.Module):
    def __init__(self,original,targets):
        super().__init__()
        self.spectral=original.spectral
        self.linear=original.linear
        self.films=nn.ModuleList([deepcopy(original.film) for _ in range(targets)])
        self.norms=nn.ModuleList([deepcopy(original.norm) for _ in range(targets)])

    def forward(self,streams,modulation):
        batch,targets,width,nt=streams.shape
        flat=streams.reshape(batch*targets,width,nt)
        local=F.silu(self.spectral(flat)+self.linear(flat)).reshape(batch,targets,width,nt)
        return torch.stack([self.norms[j](apply_film(local[:,j],self.films[j](modulation)))
                            for j in range(targets)],dim=1)


class TargetDecoupledVelocity(ConditionalVelocity):
    """Preserve original conditioning; specialize modulation/output for each goal.

    The shared input includes all3 states, and linear state weights remain shared.
    Target streams have independent FiLM and GroupNorm at each block, with no
    attention or private output feeding a different target's private stream.
    """
    def __init__(self,architecture='target-decoupled-film-v1',**kwargs):
        if architecture!='target-decoupled-film-v1':raise ValueError('invalid target-decoupled architecture')
        if any(kwargs.get(k,False) for k in ['channel_mlp_skip','explicit_physical_conditioning','scalar_geometry_film']):
            raise ValueError('decoupling experiment requires original condition path and blocks')
        if kwargs.get('n_targets',3)!=3:raise ValueError('requires Cl/Cd/CM targets')
        super().__init__(architecture='floral-filmfno-v1',**kwargs)
        targets=self.config['n_targets'];self.config['architecture']=architecture
        self.initial_films=nn.ModuleList([deepcopy(self.initial_film) for _ in range(targets)])
        self.linear_films=nn.ModuleList([_select_outputs(self.linear_film,[j,j+targets]) for j in range(targets)])
        self.heads=nn.ModuleList([_select_outputs(self.head,[j]) for j in range(targets)])
        self.blocks=nn.ModuleList([TargetModulatedBlock(block,targets) for block in self.blocks])
        del self.initial_film,self.linear_film,self.head

    def forward_branches(self,g,t,condition):
        if g.ndim!=3 or g.shape[-1]!=3:raise ValueError('state must be [batch,theta,3]')
        batch,nt,_=g.shape
        if t.shape not in ((batch,),(batch,1,1)):raise ValueError('invalid FM time shape')
        features=condition['features'];encoded=condition['encoded_features']
        if features.shape!=(batch,self.config['lf_channels']+self.config['rotor_channels']+self.config['height_width']+2,nt):
            raise ValueError('original angular condition features required')
        if encoded.shape!=(batch,self.config['width'],nt):raise ValueError('encoded condition shape mismatch')
        time=self.time_embedding(t)[:,:,None].expand(-1,-1,nt)
        modulation=torch.cat((encoded,time),dim=1)
        raw=g.transpose(1,2)
        linear_raw=self.linear_spectral(raw)+self.linear_pointwise(raw)
        linear=torch.cat([apply_film(linear_raw[:,j:j+1],self.linear_films[j](modulation))
                          for j in range(3)],dim=1)
        inputs=(raw,time,features) if self.config['direct_conditioning'] else (raw,time)
        lifted=self.lift(torch.cat(inputs,dim=1))
        streams=torch.stack([apply_film(lifted,film(modulation)) for film in self.initial_films],dim=1)
        for block in self.blocks:streams=block(streams,modulation)
        nonlinear=torch.cat([self.heads[j](streams[:,j]) for j in range(3)],dim=1)
        return linear.transpose(1,2),nonlinear.transpose(1,2)
