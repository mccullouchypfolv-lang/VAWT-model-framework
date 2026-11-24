"""Opt-in direct physical-condition FiLM path on the unchanged default backbone."""
import torch
from torch import nn
from .residual_flow import ConditionalVelocity, apply_film


class PhysicalFiLMVelocity(ConditionalVelocity):
    def __init__(self, physical_width=64, architecture='physical-film-path-v1', **options):
        if architecture!='physical-film-path-v1':
            raise ValueError('unsupported physical FiLM architecture')
        if isinstance(physical_width,bool) or not isinstance(physical_width,int) or physical_width<1:
            raise ValueError('physical_width must be a positive integer')
        if any(options.get(key,False) for key in ('explicit_physical_conditioning','scalar_geometry_film','channel_mlp_skip')):
            raise ValueError('physical FiLM experiment requires the unchanged default backbone')
        super().__init__(architecture='floral-filmfno-v1',**options)
        # Initialize additions last: all common parameters retain same-seed values.
        self.physical_encoder=nn.Sequential(
            nn.Linear(self.config['n_conditions']+40,physical_width),nn.SiLU(),
            nn.Linear(physical_width,physical_width),nn.SiLU())
        self.physical_heads=nn.ModuleList([
            nn.Linear(physical_width,2*self.config['width']) for _ in self.blocks])
        for head in self.physical_heads:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.config.update(architecture=architecture,physical_width=physical_width)

    def prepare_condition(self,lf,rotor,field,a,z,weights,theta,geometry=None):
        if geometry is None or geometry.shape!=(len(lf),20,2) or not torch.isfinite(geometry).all():
            raise ValueError('direct physical FiLM requires finite normalized geometry [B,20,2]')
        condition=super().prepare_condition(lf,rotor,field,a,z,weights,theta)
        condition['physical_encoded']=self.physical_encoder(torch.cat((a,geometry.flatten(1)),dim=1))
        return condition

    def nonlinear_block(self,index,x,modulation,condition):
        x=super().nonlinear_block(index,x,modulation,condition)
        return apply_film(x,self.physical_heads[index](condition['physical_encoded']).unsqueeze(-1))
