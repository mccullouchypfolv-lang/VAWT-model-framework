"""Independent scalar FiLM branches for the layer-wind LF operator."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import FiLMFNO,FiLMBlock,reflect_height
from ._vendor.neuralop_embeddings import GridEmbeddingND

SCALARS=('wind_ref_m_s','tsr_ref','chord_m','thickness_ratio')


class UnconditionedFNOBlock(FiLMBlock):
    def __init__(self,width,modes_z,modes_theta,pad_z):
        super().__init__(width,4,modes_z,modes_theta,pad_z)
        del self.film

    def forward(self,x):
        global_features=self.spectral(reflect_height(x,self.pad_z))
        if self.pad_z:global_features=global_features[:,:,self.pad_z:-self.pad_z]
        features=global_features+self.linear(x)+self.local(reflect_height(x,1))
        return F.gelu(features)


class SeparateScalarFiLM(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.scalar_film=nn.ModuleDict({key:nn.Sequential(nn.Linear(1,width),nn.GELU(),nn.Linear(width,2*width)) for key in SCALARS})

    def forward(self,features,scalars):
        parameters=sum(self.scalar_film[key](scalars[key]) for key in SCALARS)
        gamma,beta=parameters.chunk(2,dim=-1)
        return (1+gamma[:,:,None,None])*features+beta[:,:,None,None]


class SeparateScalarLFFNO(FiLMFNO):
    scalar_names=SCALARS

    def __init__(self,input_layout='separate-scalar-layer-wind-v1',film_placement='post_block',positional_embedding='grid-nd-angular-v1',**kwargs):
        if input_layout!='separate-scalar-layer-wind-v1':raise ValueError('unsupported separate scalar layout')
        if film_placement!='post_block':raise ValueError('separate scalar LF requires post-block FiLM')
        if positional_embedding!='grid-nd-angular-v1':raise ValueError('separate scalar LF requires angular GridEmbeddingND')
        super().__init__(input_layout='scalar-geometry-layer-wind-v1',**kwargs)
        self.config['input_layout']=input_layout
        self.config['film_placement']=film_placement
        self.config['positional_embedding']=positional_embedding
        c=self.config
        self.grid_embedding=GridEmbeddingND(in_channels=1,dim=1,grid_boundaries=[[0.,1.]])
        self.lift=nn.Conv2d(self.grid_embedding.out_channels,c['width'],1)
        self.blocks=nn.ModuleList([UnconditionedFNOBlock(c['width'],c['modes_z'],c['modes_theta'],c['pad_z']) for _ in range(c['blocks'])])
        self.modulations=nn.ModuleList([SeparateScalarFiLM(c['width']) for _ in range(c['blocks'])])

    def forward(self,x,wind_ref_m_s,tsr_ref,chord_m,thickness_ratio,theta,return_features=False):
        if x.ndim!=2 or x.shape[1]!=20:raise ValueError('wind must be [B,20]')
        b=x.shape[0]
        scalars=dict(zip(SCALARS,(wind_ref_m_s,tsr_ref,chord_m,thickness_ratio)))
        if any(v.shape!=(b,1) or not torch.isfinite(v).all() for v in scalars.values()):
            raise ValueError('each scalar condition must be finite [B,1]')
        if theta.ndim!=2 or theta.shape[0]!=b or not torch.isfinite(theta).all() or not torch.isfinite(x).all():
            raise ValueError('invalid angular grid or wind')
        n=theta.shape[1]
        expected=(torch.arange(n,device=theta.device,dtype=theta.dtype)+.5)*(2*torch.pi/n)
        if not torch.allclose(theta,expected[None].expand(b,-1),atol=1e-6,rtol=1e-6):
            raise ValueError('automatic grid requires the declared uniform midpoint angular grid')
        # Treat each height as a separate batch item for positional embedding;
        # dim=1 appends theta only, never an implicit height coordinate.
        wind=x[:,:,None].expand(-1,-1,n).reshape(b*20,1,n)
        field=self.grid_embedding(wind).reshape(b,20,2,n).permute(0,2,1,3)
        field=self.lift(field)
        for block,film in zip(self.blocks,self.modulations):
            field=film(block(field),scalars)
        rotor=field.mean(2)
        prediction=self.head(rotor).transpose(1,2)
        if return_features:return dict(prediction=prediction,field_features=field,rotor_features=rotor)
        return prediction
