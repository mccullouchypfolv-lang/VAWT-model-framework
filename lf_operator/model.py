"""Mixed-boundary 2D-to-1D FiLM Fourier neural operator."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def reflect_height(x, padding):
    """Reflect z without CUDA reflection_pad2d's nondeterministic backward."""
    if padding == 0:
        return x
    if not 0 < padding < x.shape[2]:
        raise ValueError('height reflection padding must be smaller than z grid')
    return torch.cat((x[:,:,1:padding+1].flip(2),x,x[:,:,-padding-1:-1].flip(2)),dim=2)


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes_z, modes_theta):
        super().__init__()
        if min(width,modes_z,modes_theta) < 1:
            raise ValueError('width and modes must be positive')
        self.modes_z, self.modes_theta = modes_z, modes_theta
        scale = 1/math.sqrt(width)
        self.weight_pos = nn.Parameter(scale*torch.randn(width,width,modes_z,modes_theta,dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale*torch.randn(width,width,modes_z,modes_theta,dtype=torch.cfloat))

    def forward(self, x):
        b,c,nz,nt = x.shape
        mz, mt = self.modes_z, self.modes_theta
        if 2*mz > nz or mt > nt//2+1:
            raise ValueError('spectral modes overlap or exceed grid resolution')
        spectrum = torch.fft.rfft2(x, norm='ortho')
        out = spectrum.new_zeros(b,c,nz,nt//2+1)
        out[:,:,:mz,:mt] = torch.einsum('bizt,iozt->bozt',spectrum[:,:,:mz,:mt],self.weight_pos)
        out[:,:,-mz:,:mt] = torch.einsum('bizt,iozt->bozt',spectrum[:,:,-mz:,:mt],self.weight_neg)
        return torch.fft.irfft2(out, s=(nz,nt), norm='ortho')


class FiLMBlock(nn.Module):
    def __init__(self,width,n_conditions,modes_z,modes_theta,pad_z):
        super().__init__()
        self.pad_z = pad_z
        self.spectral = SpectralConv2d(width,modes_z,modes_theta)
        self.linear = nn.Conv2d(width,width,1)
        self.local = nn.Conv2d(width,width,(3,1))
        self.film = nn.Sequential(nn.Linear(n_conditions,width),nn.GELU(),nn.Linear(width,2*width))

    def forward(self,x,a):
        pad = self.pad_z
        extended = reflect_height(x,pad)
        global_features = self.spectral(extended)
        if pad:
            global_features = global_features[:,:,pad:-pad]
        local = self.local(reflect_height(x,1))
        features = global_features+self.linear(x)+local
        gamma,beta = self.film(a).chunk(2,dim=-1)
        return F.gelu((1+gamma[:,:,None,None])*features+beta[:,:,None,None])


class FiLMFNO(nn.Module):
    def __init__(self,n_features=6,n_conditions=4,width=32,blocks=4,
                 modes_z=6,modes_theta=12,pad_z=6):
        super().__init__()
        if min(n_features,n_conditions,width,blocks) < 1 or not 0 <= pad_z < 20:
            raise ValueError('invalid model dimensions or height reflection padding')
        self.config = dict(n_features=n_features,n_conditions=n_conditions,width=width,
                           blocks=blocks,modes_z=modes_z,modes_theta=modes_theta,pad_z=pad_z)
        self.lift = nn.Conv2d(n_features+3,width,1)
        self.blocks = nn.ModuleList([FiLMBlock(width,n_conditions,modes_z,modes_theta,pad_z) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv1d(width,width,1),nn.GELU(),nn.Conv1d(width,2,1))

    def forward(self,x,a,z,theta,weights,return_features=False):
        if x.ndim != 4 or x.shape[1] != 20 or x.shape[2] != self.config['n_features']:
            raise ValueError('x must be [B,20,n_features,N_theta]')
        b,nz,_,nt = x.shape
        if a.shape != (b,self.config['n_conditions']) or z.shape != (b,nz) or theta.shape != (b,nt) or weights.shape != (b,nz):
            raise ValueError('condition/coordinate/weight shape mismatch')
        if not torch.isfinite(weights).all() or (weights<=0).any():
            raise ValueError('height weights must be finite and positive')
        coords = torch.stack((z[:,:,None].expand(-1,-1,nt),
                              theta.sin()[:,None,:].expand(-1,nz,-1),
                              theta.cos()[:,None,:].expand(-1,nz,-1)),dim=1)
        field = self.lift(torch.cat((x.permute(0,2,1,3),coords),dim=1))
        for block in self.blocks:
            field = block(field,a)
        weights = weights/weights.sum(-1,keepdim=True)
        rotor_features = (field*weights[:,None,:,None]).sum(2)
        prediction = self.head(rotor_features).transpose(1,2)
        if return_features:
            return dict(prediction=prediction,field_features=field,rotor_features=rotor_features)
        return prediction
