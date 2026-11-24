"""Full-pass condition batches and FLORAL's stochastic linear bridge."""
import numpy as np
import torch
from torch.utils.data import DataLoader


def epoch_indices(train_indices,*,batch_size,seed,epoch):
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('batch_size must be a positive integer')
    ids=np.asarray(train_indices,dtype=np.int64)
    if ids.ndim!=1 or not len(ids) or len(np.unique(ids))!=len(ids):raise ValueError('distinct training indices required')
    generator=torch.Generator().manual_seed(int(seed)+int(epoch))
    loader=DataLoader(torch.as_tensor(ids),batch_size=batch_size,shuffle=True,drop_last=False,
                      generator=generator,num_workers=0)
    for batch in loader:yield batch.numpy()


def validate_bridge(cfg):
    if cfg is None:return
    if cfg.get('method')!='floral':raise ValueError('unsupported flow bridge')
    if set(cfg)-{'method','sig_min','prior_covariance_factor'}:raise ValueError('unknown flow bridge options')
    for key,default in [('sig_min',1e-6),('prior_covariance_factor',2.)]:
        value=cfg.get(key,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0 or (key=='prior_covariance_factor' and value==0):
            raise ValueError('invalid bridge '+key)


def base_bridge_noise(source,count,rng,*,prior_covariance_factor=2.):
    from .blade_pipeline import sample_multichannel_source
    if not np.isfinite(prior_covariance_factor) or prior_covariance_factor<=0:raise ValueError('invalid covariance ratio')
    # NAGO fits effective residual-prior covariance P. Official prior=2*C_base;
    # hence bridge GP covariance is C_base=P/2, without changing the fitted P.
    return (sample_multichannel_source(source,count,rng)/np.sqrt(prior_covariance_factor)).astype('float32')


def flow_path(g0,target,t,source,rng,bridge=None):
    validate_bridge(bridge)
    state=(1-t)*g0+t*target
    if bridge is not None:
        noise=base_bridge_noise(source,len(g0),rng,prior_covariance_factor=bridge.get('prior_covariance_factor',2.))
        mse=np.mean((target-g0)**2,axis=tuple(range(1,target.ndim)),keepdims=True)
        state=state+bridge.get('sig_min',1e-6)*mse*noise
    return state.astype(g0.dtype,copy=False)
