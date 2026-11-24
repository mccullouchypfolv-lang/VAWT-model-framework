"""Unified unpooled physical conditions, independent of frozen LF latents."""
import numbers
import torch
from torch import nn
from .residual_flow import SpectralConv1d,VelocityBlock,FourierTimeEmbedding,film_network,apply_film


class RawConditionVelocity(nn.Module):
    """State/time lifting and separate physical-condition lifting with FiLM.

    Retains original project's FNO blocks/linear spectral operator. Aligns
    condition routing, not every block of the official FLORAL dependency.
    """
    def __init__(self,width=64,blocks=4,modes=6,lift_ratio=4,projection_ratio=4,
                 time_embedding_dim=32,n_targets=3,lf_channels=3,n_conditions=2,
                 architecture='raw-physical-film-v1',direct_conditioning=False,
                 explicit_physical_conditioning=True):
        super().__init__()
        dimensions=dict(width=width,blocks=blocks,modes=modes,lift_ratio=lift_ratio,
            projection_ratio=projection_ratio,time_embedding_dim=time_embedding_dim,
            n_targets=n_targets,lf_channels=lf_channels,n_conditions=n_conditions)
        if any(isinstance(v,bool) or not isinstance(v,numbers.Integral) or v<1 for v in dimensions.values()):
            raise ValueError('positive integer dimensions required')
        if architecture!='raw-physical-film-v1' or direct_conditioning or not explicit_physical_conditioning:
            raise ValueError('raw physical conditions require separate FiLM routing')
        if n_conditions!=2 or time_embedding_dim<2 or time_embedding_dim%2:
            raise ValueError('requires U/TSR and even time embedding dimension')
        self.config=dict(dimensions,architecture=architecture,direct_conditioning=False,
                         explicit_physical_conditioning=True)
        # No LF curve, frozen rotor features or height pooling in this branch.
        self.condition_encoder=nn.Sequential(nn.Conv1d(64,width*lift_ratio,1),nn.SiLU(),
                                             nn.Conv1d(width*lift_ratio,width,1))
        self.time_embedding=FourierTimeEmbedding(time_embedding_dim)
        self.lift=nn.Sequential(nn.Conv1d(n_targets+time_embedding_dim,width*lift_ratio,1),
                                nn.SiLU(),nn.Conv1d(width*lift_ratio,width,1))
        self.initial_film=film_network(width,width)
        self.linear_spectral=SpectralConv1d(n_targets,modes)
        self.linear_pointwise=nn.Conv1d(n_targets,n_targets,1)
        self.linear_film=film_network(width,n_targets)
        self.blocks=nn.ModuleList([VelocityBlock(width,modes,width) for _ in range(blocks)])
        self.head=nn.Sequential(nn.Conv1d(width,width*projection_ratio,1),nn.SiLU(),
                               nn.Conv1d(width*projection_ratio,n_targets,1))

    def prepare_condition(self,lf,rotor,field,a,z,weights,theta,geometry=None):
        # Legacy arguments are accepted for pipeline compatibility, never read.
        if geometry is None or geometry.ndim!=3 or geometry.shape[1:]!=(20,2):
            raise ValueError('complete normalized geometry [B,20,2] required')
        batch=geometry.shape[0]
        if a.shape!=(batch,2) or z.shape!=(batch,20) or theta.ndim!=2 or theta.shape[0]!=batch:
            raise ValueError('physical condition shape mismatch')
        if not all(torch.isfinite(x).all() for x in (geometry,a,z,theta)):
            raise ValueError('physical conditions must be finite')
        profile=torch.cat((z[:,:,None],geometry),dim=-1).reshape(batch,60)
        physical=torch.cat((profile,a),dim=-1)[:,:,None].expand(-1,-1,theta.shape[1])
        condition=torch.cat((physical,theta.sin()[:,None],theta.cos()[:,None]),dim=1)
        return dict(raw_condition=condition,encoded_features=self.condition_encoder(condition))

    def forward_branches(self,g,t,condition):
        if g.ndim!=3 or g.shape[-1]!=self.config['n_targets']:
            raise ValueError('state must be [B,theta,targets]')
        batch,nt,_=g.shape
        if t.shape not in ((batch,),(batch,1,1)):raise ValueError('invalid FM time shape')
        encoded=condition['encoded_features']
        if encoded.shape!=(batch,self.config['width'],nt):raise ValueError('condition field shape mismatch')
        raw=g.transpose(1,2)
        time=self.time_embedding(t)[:,:,None].expand(-1,-1,nt)
        linear=apply_film(self.linear_spectral(raw)+self.linear_pointwise(raw),self.linear_film(encoded))
        hidden=apply_film(self.lift(torch.cat((raw,time),dim=1)),self.initial_film(encoded))
        for block in self.blocks:hidden=block(hidden,encoded)
        return linear.transpose(1,2),self.head(hidden).transpose(1,2)

    def forward(self,g,t,condition):
        linear,nonlinear=self.forward_branches(g,t,condition)
        return linear+nonlinear
