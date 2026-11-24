"""Independent FiLM path for scalar geometry and uninduced layer inflows."""
import torch
from torch import nn
from .residual_flow import ConditionalVelocity,apply_film


class LayerWindVelocity(ConditionalVelocity):
    def __init__(self,physical_width=64,architecture='layer-wind-film-path-v1',**options):
        if architecture!='layer-wind-film-path-v1':raise ValueError('unsupported layer wind architecture')
        if isinstance(physical_width,bool) or not isinstance(physical_width,int) or physical_width<1:
            raise ValueError('physical_width must be a positive integer')
        if any(options.get(key,False) for key in ('explicit_physical_conditioning','scalar_geometry_film','channel_mlp_skip')):
            raise ValueError('layer wind experiment requires the unchanged default backbone')
        super().__init__(architecture='floral-filmfno-v1',**options)
        self.physical_encoder=nn.Sequential(nn.Linear(self.config['n_conditions']+2+20,physical_width),
            nn.SiLU(),nn.Linear(physical_width,physical_width),nn.SiLU())
        self.physical_heads=nn.ModuleList([nn.Linear(physical_width,2*self.config['width']) for _ in self.blocks])
        for head in self.physical_heads:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.config.update(architecture=architecture,physical_width=physical_width)

    def prepare_condition(self,lf,rotor,field,a,z,weights,theta,scalar_geometry=None,layer_wind=None):
        batch=len(lf)
        if scalar_geometry is None or scalar_geometry.shape!=(batch,2) or not torch.isfinite(scalar_geometry).all():
            raise ValueError('scalar chord/thickness inputs must be finite [B,2]')
        if layer_wind is None or layer_wind.shape!=(batch,20) or not torch.isfinite(layer_wind).all():
            raise ValueError('uninduced layer winds must be finite [B,20]')
        condition=super().prepare_condition(lf,rotor,field,a,z,weights,theta)
        condition['physical_encoded']=self.physical_encoder(torch.cat((a,scalar_geometry,layer_wind),dim=1))
        return condition

    def nonlinear_block(self,index,x,modulation,condition):
        return apply_film(super().nonlinear_block(index,x,modulation,condition),
            self.physical_heads[index](condition['physical_encoded']).unsqueeze(-1))
