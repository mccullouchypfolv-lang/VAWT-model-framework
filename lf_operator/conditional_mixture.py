"""Condition-dependent periodic Gaussian-mixture centers, with fixed covariance."""
import numpy as np
import torch
from torch import nn


class ConditionalMixtureSource(nn.Module):
    def __init__(self,components=2,modes=5,width=64,n_targets=3):
        super().__init__()
        for v in (components,modes,width,n_targets):
            if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError('positive integer source dimensions required')
        self.config=dict(components=components,modes=modes,width=width,n_targets=n_targets)
        self.n_coefficients=1+2*modes
        descriptor=60+2+n_targets*self.n_coefficients
        self.encoder=nn.Sequential(nn.Linear(descriptor,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.centers=nn.Linear(width,components*n_targets*self.n_coefficients)
        self.logits=nn.Linear(width,components)
        nn.init.normal_(self.centers.weight,std=1e-3);nn.init.zeros_(self.centers.bias)
        nn.init.zeros_(self.logits.weight);nn.init.zeros_(self.logits.bias)

    def forward(self,geometry,z,operating,lf,theta):
        batch,nt,channels=lf.shape
        if channels!=self.config['n_targets'] or geometry.shape!=(batch,20,2) or z.shape!=(batch,20) or operating.shape!=(batch,2) or theta.shape!=(batch,nt):
            raise ValueError('invalid conditional mixture inputs')
        if nt<=2*self.config['modes']:raise ValueError('angular grid must resolve all source mean harmonics')
        k=torch.arange(1,self.config['modes']+1,device=lf.device,dtype=lf.dtype)
        phase=theta[:,None,:]*k[None,:,None]
        basis=torch.cat((torch.ones_like(theta[:,None,:]),phase.cos(),phase.sin()),dim=1)
        lf_coefficients=torch.einsum('bnc,bln->bcl',lf,basis)/nt
        lf_coefficients=lf_coefficients*lf.new_tensor([1.]+[2.]*(self.n_coefficients-1))
        profile=torch.cat((geometry,z[:,:,None]),dim=-1).reshape(batch,60)
        descriptor=torch.cat((profile,operating,lf_coefficients.reshape(batch,-1)),dim=1)
        hidden=self.encoder(descriptor)
        coefficients=self.centers(hidden).reshape(batch,self.config['components'],channels,self.n_coefficients)
        means=torch.einsum('bmcl,bln->bmnc',coefficients,basis)
        return means,self.logits(hidden).softmax(dim=-1)


def expected_geodesic_loss(means,probabilities,target,noise_variance=0.):
    """Exact E||g0-r||² for one target per condition, including fixed noise cost."""
    if means.ndim!=4 or target.shape!=(means.shape[0],means.shape[2],means.shape[3]) or probabilities.shape!=means.shape[:2]:
        raise ValueError('invalid mixture/target shapes')
    if not np.isfinite(noise_variance) or noise_variance<0:raise ValueError('invalid fixed noise variance')
    costs=(means-target[:,None]).square().mean((2,3))
    return (costs*probabilities).sum(1).mean()+noise_variance


def select_mixture_centers(means,probabilities,uniforms):
    if probabilities.shape!=means.shape[:2] or uniforms.shape!=(means.shape[0],):raise ValueError('invalid categorical draw shapes')
    if not torch.isfinite(probabilities).all() or (probabilities<0).any() or not torch.allclose(probabilities.sum(1),torch.ones_like(uniforms),atol=1e-5):
        raise ValueError('mixture probabilities must be normalized')
    if ((uniforms<0)|(uniforms>=1)).any():raise ValueError('categorical uniforms must lie in [0,1)')
    indices=(uniforms[:,None]>=probabilities.cumsum(1)).sum(1).clamp_max(means.shape[1]-1)
    return means[torch.arange(len(means),device=means.device),indices],indices


def source_parameters(model,tensors,ids):
    return model(*(tensors[k][ids] for k in ('geometry','z','conditions','lf_prediction','theta')))


def fixed_noise_variance(source):
    total=[]
    for channel in source['channels']:
        n=channel['n_theta'];p=np.asarray(channel['psd']);m=np.full(len(p),2.);m[0]=1.
        if n%2==0:m[-1]=1.
        total.append(float(np.sum(m*p)))
    return float(np.mean(total))
