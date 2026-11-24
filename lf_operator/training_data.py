"""Load parameter-only training inputs, keeping labels outside the forward API."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


def compact_lf_inputs(x,a,z,theta,stats,config,*,separate=False):
    """Scalar geometry and prescribed wind only; height never enters the model.

    One training-fitted reference-wind scale is shared by all heights, preserving
    the prescribed shear profile. No induced or relative flow is read.
    """
    x,a,z,theta=[np.asarray(v,dtype=float) for v in (x,a,z,theta)]
    if x.shape!=(20,2,len(theta)) or a.shape!=(2,) or z.shape!=(20,):
        raise ValueError('invalid compact LF raw input dimensions')
    if not all(np.isfinite(v).all() for v in (x,a,z,theta)) or np.any(a<=0) or np.any(z<=0):
        raise ValueError('invalid physical compact LF inputs')
    if not np.allclose(x,x[:1,:,:1],rtol=1e-6,atol=1e-6):
        raise ValueError('compact LF requires constant chord/thickness across height and azimuth')
    geometry=x[0,:,0]
    if geometry[0]<=0 or not 0<geometry[1]<1:raise ValueError('invalid scalar geometry')
    profile=config['wind_profile']
    if profile=='power_law':
        wind=a[0]*(z/float(config['wind_reference_height']))**float(config['shear_exponent'])
    elif profile=='uniform':wind=np.full(20,a[0])
    else:raise ValueError('unsupported prescribed wind profile')
    am,ast=np.asarray(stats['a_mean']),np.asarray(stats['a_std'])
    gm,gst=np.asarray(stats['x_mean']),np.asarray(stats['x_std'])
    if np.any(ast<=0) or np.any(gst<=0):raise ValueError('invalid training input scales')
    result=dict(x=(wind-am[0])/ast[0],theta=theta)
    if separate:
        from .separate_lf import SCALARS
        conditions=(a-am)/ast;geom=(geometry-gm)/gst
        result.update({key:np.array([value]) for key,value in zip(SCALARS,(*conditions,*geom))})
    else:result['a']=np.concatenate(((a-am)/ast,(geometry-gm)/gst))
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite compact LF input')
    return {k:v.astype('float32') for k,v in result.items()}


class LFData(Dataset):
    def __init__(self,cache,manifest,indices,stats,require_targets=True,input_layout=None):
        self.cache=Path(cache); self.manifest=manifest; self.indices=indices
        self.stats=stats; self.require_targets=require_targets
        self.input_layout=input_layout

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        config=self.manifest['config']; stats=self.stats
        with np.load(self.cache/self.manifest['records'][self.indices[index]]['file'],allow_pickle=False) as saved:
            x=np.asarray(saved['x'],dtype='float32'); a=np.asarray(saved['a'],dtype='float32')
            if x.ndim!=3 or x.shape[:2]!=(20,2) or a.shape!=(2,):raise ValueError('expected two geometry channels and [U_ref,TSR]')
            result=dict(x=(x-np.asarray(stats['x_mean'],dtype='float32')[None,:,None])/np.asarray(stats['x_std'],dtype='float32')[None,:,None],
                        a=(a-np.asarray(stats['a_mean'],dtype='float32'))/np.asarray(stats['a_std'],dtype='float32'),
                        z=(np.asarray(saved['z'],dtype='float32')-config['rotor_bottom_height'])/config['height'],
                        theta=np.asarray(saved['theta'],dtype='float32'),weights=np.asarray(saved['weights'],dtype='float32'))
            if self.input_layout in ('scalar-geometry-layer-wind-v1','separate-scalar-layer-wind-v1'):
                result=compact_lf_inputs(x,a,saved['z'],saved['theta'],stats,config,separate=self.input_layout=='separate-scalar-layer-wind-v1')
            if self.require_targets:
                target=np.asarray(saved['y'],dtype='float32')
                result['y']=(target-np.asarray(stats['y_mean'],dtype='float32'))/np.asarray(stats['y_std'],dtype='float32')
        return {k:torch.from_numpy(v) for k,v in result.items()}


def loader(cache,manifest,indices,stats,batch_size,shuffle=False,require_targets=True,input_layout=None):
    generator=None if shuffle else torch.Generator().manual_seed(0)
    return DataLoader(LFData(cache,manifest,indices,stats,require_targets,input_layout),batch_size=batch_size,
                      shuffle=shuffle,num_workers=0,generator=generator)


def to_device(batch,device):return {k:v.to(device) for k,v in batch.items()}


def forward(model,batch,features=False):
    if model.config.get('input_layout')=='separate-scalar-layer-wind-v1':
        return model(batch['x'],**{k:batch[k] for k in (*model.scalar_names,'theta')},return_features=features)
    if model.config.get('input_layout')=='scalar-geometry-layer-wind-v1':
        return model(batch['x'],batch['a'],theta=batch['theta'],return_features=features)
    return model(*(batch[k] for k in ('x','a','z','theta','weights')),return_features=features)
