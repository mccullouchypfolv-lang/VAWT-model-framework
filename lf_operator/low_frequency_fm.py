"""Train-only residual-energy weighting of complex low-frequency FM errors."""
import numpy as np
import torch
from .objectives import validate_time_weighting


def fit_low_frequency_weights(values,indices,stats,options):
    from .blade_pipeline import encode_residual
    if not isinstance(options,dict) or set(options)-{'energy_fraction','energy_power','strength'}:
        raise ValueError('invalid low-frequency FM options')
    fraction=options.get('energy_fraction',.95);power=options.get('energy_power',.5);strength=options.get('strength',1.)
    for name,value in [('energy_fraction',fraction),('energy_power',power),('strength',strength)]:
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value):
            raise ValueError(name+' must be finite')
    if not 0<fraction<=1 or not 0<=power<=1 or strength<0:raise ValueError('invalid spectral weighting range')
    ids=np.asarray(indices,dtype=int)
    if ids.ndim!=1 or len(ids)<2 or len(np.unique(ids))!=len(ids) or ids.tolist()!=stats['fit_indices']:
        raise ValueError('spectral weights must use normalization training indices only')
    residual=encode_residual(np.asarray(values)[ids],stats)
    if residual.ndim!=3 or not np.isfinite(residual).all():raise ValueError('finite training residual curves required')
    n=residual.shape[1];psd=np.mean(abs(np.fft.rfft(residual,axis=1,norm='forward'))**2,axis=0)
    multiplicity=np.full(len(psd),2.);multiplicity[0]=1.
    if n%2==0:multiplicity[-1]=1.
    energy=psd*multiplicity[:,None];total=energy.sum(0)
    if (total<=0).any():raise ValueError('nondegenerate residual channels required')
    cumulative=np.cumsum(energy,axis=0)/total
    weights=np.zeros_like(psd);cutoffs=[];retained=[]
    for c in range(psd.shape[1]):
        cutoff=min(int(np.searchsorted(cumulative[:,c],fraction)),len(psd)-1)
        band=psd[:cutoff+1,c];m=multiplicity[:cutoff+1]
        relative=np.maximum(band,max(float(band.max())*1e-8,np.finfo(float).tiny))**power
        weights[:cutoff+1,c]=relative/(np.sum(relative*m)/m.sum())
        cutoffs.append(cutoff);retained.append(float(cumulative[cutoff,c]))
    return dict(fit_indices=ids.tolist(),n_theta=n,energy_fraction=fraction,energy_power=power,
        strength=strength,cutoff_modes=cutoffs,retained_energy_fraction=retained,weights=weights.tolist(),
        residual_psd=psd.tolist(),coordinate_space='training-normalized HF minus LF prediction residual',
        fft_norm='forward',weight_normalization='multiplicity-weighted mean one within each channel band')


def low_frequency_velocity_loss(prediction,target,weights,*,time=None,time_weighting='none'):
    """Weighted complex coefficient MSE; rFFT multiplicity preserves Parseval."""
    validate_time_weighting(time_weighting)
    if prediction.ndim!=3 or prediction.shape!=target.shape:raise ValueError('matching [batch,theta,channel] required')
    error=prediction-target;n=error.shape[1]
    spectrum=torch.fft.rfft(error,dim=1,norm='forward')
    w=torch.as_tensor(weights,dtype=error.dtype,device=error.device)
    if w.shape!=spectrum.shape[1:] or not torch.isfinite(w).all() or (w<0).any() or (w.sum(0)<=0).any():
        raise ValueError('finite nonnegative frequency weights required for every channel')
    multiplicity=error.new_full((spectrum.shape[1],),2.);multiplicity[0]=1.
    if n%2==0:multiplicity[-1]=1.
    per_sample=(spectrum.abs().square()*w*multiplicity[:,None]).sum(1).mean(1)
    if time_weighting=='floral_quadratic':
        if time is None:raise ValueError('time required for quadratic weighting')
        t=torch.as_tensor(time,dtype=error.dtype,device=error.device)
        if t.ndim<1 or t.shape[0]!=error.shape[0] or t.numel()!=error.shape[0] or not torch.isfinite(t).all() or ((t<0)|(t>1)).any():
            raise ValueError('one finite time in [0,1] per sample required')
        per_sample=per_sample*(1+2*t.reshape(-1).square())
    return per_sample.mean()
