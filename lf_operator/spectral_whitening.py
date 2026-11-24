"""Train-only reversible modal preconditioning of real periodic residuals.

The FNO still sees an angular field. Its orthonormal Fourier coefficients are
whitened. This is not the manuscript's SpectralDiT implementation.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np


def fit_whitening(values,indices,*,variance_floor_ratio=1e-6):
    from .blade_pipeline import residual_normalization
    if not np.isfinite(variance_floor_ratio) or not 0<variance_floor_ratio<=1:
        raise ValueError('positive relative variance floor in (0,1] required')
    ids=np.asarray(indices,dtype=int);values=np.asarray(values,dtype=float)
    if ids.ndim!=1 or len(ids)<2 or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(values)):
        raise ValueError('distinct valid training cases required')
    stats=residual_normalization(values,ids.tolist());n=values.shape[1]
    centered=values[ids]-np.asarray(stats['mean_profile'])
    q=np.fft.rfft(centered,axis=1,norm='ortho');variance=np.mean(abs(q)**2,axis=0)
    floor=variance.max(axis=0)*variance_floor_ratio
    scale=np.sqrt(np.maximum(variance,floor))
    stats['spectral_whitening']=dict(reference_n_theta=n,fft_norm='ortho',modal_scale=scale.tolist(),
        variance_floor_ratio=variance_floor_ratio,floored_fraction=np.mean(variance<floor,axis=0).tolist(),
        raw_variance_ratio=(variance.max(axis=0)/np.maximum(variance.min(axis=0),np.finfo(float).tiny)).tolist(),
        whitened_modal_variance=(variance/scale**2).tolist(),
        scope='Diagonal modal variance normalization, not full cross-mode covariance whitening.')
    return stats


def whiten(values,stats):
    x=np.asarray(values,dtype=float);n=stats['spectral_whitening']['reference_n_theta']
    if x.shape[-2:]!=(n,3) or not np.isfinite(x).all():raise ValueError('native finite residual curves required')
    q=np.fft.rfft(x-np.asarray(stats['mean_profile']),axis=-2,norm='ortho')
    q=q/np.asarray(stats['spectral_whitening']['modal_scale'])
    return np.fft.irfft(q,n=n,axis=-2,norm='ortho').astype('float32')


def recolor(values,stats):
    z=np.asarray(values,dtype=float);n=z.shape[-2]
    ref=stats['spectral_whitening']['reference_n_theta']
    if z.shape[-1]!=3 or n<ref or not np.isfinite(z).all():
        raise ValueError('finite encoded curves on native or finer angular grids required')
    scale=np.asarray(stats['spectral_whitening']['modal_scale'])
    # A real Fourier multiplier commutes with grid-origin phase shifts. Forward
    # normalization preserves coefficient amplitude under angular refinement.
    q=np.fft.rfft(z,axis=-2,norm='forward');out=np.zeros_like(q)
    out[...,:len(scale),:]=q[...,:len(scale),:]*scale
    return np.fft.irfft(out,n=n,axis=-2,norm='forward')+np.asarray(stats['mean_profile'])


def white_reference(stats,original):
    source=deepcopy(original);n=stats['spectral_whitening']['reference_n_theta']
    # Existing sampler uses forward covariance; 1/N corresponds to unit real
    # angular variance and unit orthonormal modal variance (also DC/Nyquist).
    source['channels']=[dict(kind='whitened_white',n_theta=n,n_channels=1,fft_norm='forward',
        domain=[0.,float(2*np.pi)],psd=[1/n]*(n//2+1)) for _ in range(3)]
    source['coordinate_system']='spectrally whitened residual field'
    source['source_policy']='unit white reference in whitened coordinates; replaces NAGO for this experiment'
    source['cross_channel_covariance']='independent source channels; independent conditional coupling'
    return source


def prepare_whitened_assets(cfg):
    from .blade_pipeline import ROOT,read_json,save_json,digest,validate_flow_assets
    work=ROOT/cfg['work'];base=work/'stage2';name=cfg['flow_assets'];out=work/name
    if Path(name).name!=name or name in ('stage1','stage2','lf_cache'):raise ValueError('distinct assets directory required')
    if out.exists():raise FileExistsError(out)
    if read_json(base/'status.json')['status']!='complete':raise ValueError('original stage2 incomplete')
    provenance=read_json(base/'provenance.json')
    if not all(digest(Path(p))==h for p,h in provenance.items()):raise ValueError('original assets changed')
    with np.load(base/'paired_data.npz') as f:arrays={k:f[k] for k in f.files}
    splits=read_json(base/'splits.json');original=read_json(base/'source.json')
    validate_flow_assets(arrays,splits,read_json(base/'normalization.json'),original)
    stats=fit_whitening(arrays['residual'],splits['train'],**cfg['spectral_whitening'])
    source=white_reference(stats,original)
    z=whiten(arrays['residual'][splits['train']],stats)
    error=float(np.max(abs(recolor(z,stats)-arrays['residual'][splits['train']])))
    out.mkdir();np.savez_compressed(out/'paired_data.npz',**arrays)
    for filename,data in [('splits.json',splits),('normalization.json',stats),('source.json',source),
                          ('split_design.json',read_json(base/'split_design.json'))]:save_json(out/filename,data)
    paths=[base/'provenance.json',work/'stage1/best.pt']+[out/p for p in
        ['paired_data.npz','splits.json','normalization.json','source.json','split_design.json']]
    save_json(out/'provenance.json',{str(p.resolve()):digest(p) for p in paths})
    report=dict(status='complete',fit_indices=splits['train'],coordinate_system=source['coordinate_system'],
        source_policy=source['source_policy'],spectral_whitening=stats['spectral_whitening'],
        max_train_roundtrip_error=error,validation='unchanged original global channel scale')
    save_json(out/'status.json',report)
    return report
