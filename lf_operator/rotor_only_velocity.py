"""Original dual-branch velocity with integrated rotor conditions only."""
import torch
from torch import nn
from .residual_flow import ConditionalVelocity,apply_film


class RotorOnlyVelocity(ConditionalVelocity):
    def __init__(self,width=64,blocks=4,modes=6,rotor_channels=32,n_conditions=2,
                 n_targets=3,lf_channels=3,lift_ratio=4,projection_ratio=4,time_embedding_dim=32,
                 direct_conditioning=True,time_max_frequency=1000.,architecture='rotor-only-filmfno-v1'):
        if architecture!='rotor-only-filmfno-v1':raise ValueError('invalid rotor-only velocity architecture')
        super().__init__(width=width,blocks=blocks,modes=modes,rotor_channels=rotor_channels,n_conditions=n_conditions,
            n_targets=n_targets,lf_channels=lf_channels,lift_ratio=lift_ratio,projection_ratio=projection_ratio,
            time_embedding_dim=time_embedding_dim,direct_conditioning=direct_conditioning,time_max_frequency=time_max_frequency)
        retained=lf_channels+rotor_channels;old_height=self.config['height_width']
        columns=list(range(retained))+list(range(retained+old_height,retained+old_height+2+n_conditions))
        previous=self.condition_encoder[0];replacement=nn.Conv1d(len(columns),previous.out_channels,1)
        with torch.no_grad():
            replacement.weight.copy_(previous.weight[:,columns]);replacement.bias.copy_(previous.bias)
        self.condition_encoder[0]=replacement
        if direct_conditioning:
            prefix=n_targets+time_embedding_dim
            columns=list(range(prefix+retained))+list(range(prefix+retained+old_height,prefix+retained+old_height+2))
            previous=self.lift[0];replacement=nn.Conv1d(len(columns),previous.out_channels,1)
            with torch.no_grad():
                replacement.weight.copy_(previous.weight[:,columns]);replacement.bias.copy_(previous.bias)
            self.lift[0]=replacement
        del self.height_lift
        self.config=dict(width=width,blocks=blocks,modes=modes,rotor_channels=rotor_channels,n_conditions=n_conditions,
            n_targets=n_targets,lf_channels=lf_channels,lift_ratio=lift_ratio,projection_ratio=projection_ratio,
            time_embedding_dim=time_embedding_dim,direct_conditioning=direct_conditioning,time_max_frequency=time_max_frequency,
            architecture=architecture)

    def prepare_condition(self,lf,rotor,a,theta):
        if lf.ndim!=3 or lf.shape[-1]!=self.config['lf_channels']:raise ValueError('LF must be [batch,theta,channels]')
        batch,nt,_=lf.shape
        if rotor.shape!=(batch,self.config['rotor_channels'],nt) or a.shape!=(batch,self.config['n_conditions']) or theta.shape!=(batch,nt):
            raise ValueError('rotor-only condition shapes differ')
        features=torch.cat((lf.transpose(1,2),rotor,theta.sin()[:,None],theta.cos()[:,None]),dim=1)
        operating=a[:,:,None].expand(-1,-1,nt)
        return dict(features=features,modulation=a,encoded_features=self.condition_encoder(torch.cat((features,operating),dim=1)))

    def forward_branches(self,g,t,condition):
        if g.ndim!=3 or g.shape[-1]!=self.config['n_targets']:raise ValueError('state must be [batch,theta,targets]')
        batch,nt,_=g.shape
        if t.shape not in ((batch,),(batch,1,1)):raise ValueError('invalid time shape')
        features=condition['features'];encoded=condition['encoded_features']
        if features.shape!=(batch,self.config['lf_channels']+self.config['rotor_channels']+2,nt):raise ValueError('invalid rotor condition field')
        if encoded.shape!=(batch,self.config['width'],nt):raise ValueError('invalid encoded condition')
        time=self.time_embedding(t)[:,:,None].expand(-1,-1,nt)
        modulation=torch.cat((encoded,time),dim=1);raw=g.transpose(1,2)
        linear=apply_film(self.linear_spectral(raw)+self.linear_pointwise(raw),self.linear_film(modulation))
        inputs=(raw,time,features) if self.config['direct_conditioning'] else (raw,time)
        x=apply_film(self.lift(torch.cat(inputs,dim=1)),self.initial_film(modulation))
        for block in self.blocks:x=block(x,modulation)
        return linear.transpose(1,2),self.head(x).transpose(1,2)
