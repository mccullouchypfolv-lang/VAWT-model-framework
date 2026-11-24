"""Opt-in train-only local residual coordinates and source DC compression."""
from copy import deepcopy
from pathlib import Path
import numpy as np


def pointwise_normalization(values,indices,*,floor_ratio=.25):
    from .blade_pipeline import residual_normalization
    if not np.isfinite(floor_ratio) or not 0<floor_ratio<=1:
        raise ValueError('floor_ratio must be in (0,1]')
    base=residual_normalization(values,indices)
    train=np.asarray(values,dtype=float)[indices]
    local=np.sqrt(np.mean((train-np.asarray(base['mean_profile']))**2,axis=0))
    floor=floor_ratio*np.asarray(base['scale'])
    return dict(base,scale=np.maximum(local,floor).tolist(),validation_scale=base['scale'],
                scale_kind='pointwise',floor_ratio=floor_ratio,
                floored_fraction=np.mean(local<floor,axis=0).tolist())


def compress_source_dc(spec,factor):
    if not np.isfinite(factor) or not 0<factor<=1:
        raise ValueError('DC variance factor must be in (0,1]')
    result=deepcopy(spec);p=np.asarray(result['psd'],dtype=float)
    original=float(p[0]);p[0]*=factor
    target=np.asarray(result['target_psd'],dtype=float)
    weights=np.full(len(p),2.);weights[0]=1.
    if result['n_theta']%2==0:weights[-1]=1.
    d=result.setdefault('diagnostics',{})
    floor=float(d.get('target_psd_floor',1e-12))
    mu=np.maximum(target,floor)/np.maximum(p,floor)
    d.update(dc_variance_factor=factor,original_dc_variance=original,
        source_total_variance=float(weights@p),target_total_variance=float(weights@target),
        total_variance_difference=float(weights@(p-target)),mu=mu.tolist(),
        linear_bridge_L=(.5*(np.sqrt(mu)+1/np.sqrt(mu))).tolist(),
        dc_fraction=float(p[0]/(weights@p)))
    result['psd']=p.tolist()
    return result


def prepare_pointwise_dc(cfg):
    from .blade_pipeline import ROOT,read_json,save_json,digest,encode_residual,validate_flow_assets,TARGETS
    from .spectral_source import fit_source
    work=ROOT/cfg['work'];base=work/'stage2';name=cfg['flow_assets']
    if Path(name).name!=name or name in ('stage1','stage2','lf_cache'):
        raise ValueError('new stage2 assets must have a distinct local directory name')
    out=work/name
    if out.exists():raise FileExistsError(out)
    if read_json(base/'status.json')['status']!='complete':raise ValueError('original stage2 incomplete')
    provenance=read_json(base/'provenance.json')
    if not all(digest(Path(p))==h for p,h in provenance.items()):raise ValueError('original assets changed')
    with np.load(base/'paired_data.npz') as f:arrays={k:f[k] for k in f.files}
    splits=read_json(base/'splits.json');original=read_json(base/'source.json')
    validate_flow_assets(arrays,splits,read_json(base/'normalization.json'),original)
    stats=pointwise_normalization(arrays['residual'],splits['train'],floor_ratio=cfg['local_scale_floor_ratio'])
    encoded=encode_residual(arrays['residual'],stats)
    out.mkdir();save_json(out/'normalization.json',stats);save_json(out/'splits.json',splits)
    # Exact original physical data; never rerun LF or alter labels.
    np.savez_compressed(out/'paired_data.npz',**arrays)
    channels=[]
    try:
        for j,name in enumerate(TARGETS):
            fitted=fit_source(encoded[splits['train'],:,j:j+1],seed=cfg['seed'],**cfg['source_fit'])
            channels.append(compress_source_dc(fitted,cfg['source_dc_variance_factor']))
    except ValueError as exc:
        save_json(out/'status.json',dict(status='source_infeasible',reason=str(exc),completed_channels=len(channels)))
        raise
    source=dict(original,channels=channels)
    save_json(out/'source.json',source)
    save_json(out/'split_design.json',read_json(base/'split_design.json'))
    paths=[base/'provenance.json',work/'stage1/best.pt']+[out/p for p in
        ['paired_data.npz','splits.json','normalization.json','source.json','split_design.json']]
    save_json(out/'provenance.json',{str(p.resolve()):digest(p) for p in paths})
    report=dict(status='complete',fit_indices=splits['train'],source_dc_variance_factor=cfg['source_dc_variance_factor'],
        local_scale_floor_ratio=cfg['local_scale_floor_ratio'],beta=[c['beta'] for c in channels],
        nu=[c['nu'] for c in channels],dc_fraction=[c['diagnostics']['dc_fraction'] for c in channels],
        validation='original global channel scale; no validation/test labels fitted')
    save_json(out/'status.json',report)
    return report
