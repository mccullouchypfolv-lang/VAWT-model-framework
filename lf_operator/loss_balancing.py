"""Fixed train-only loss weights; residual/source coordinates remain unchanged."""
import numpy as np


def fit_loss_weights(residual,conditions,train_indices,stats,options):
    ids=np.asarray(train_indices,dtype=int)
    values=np.asarray(residual,dtype=float)
    a=np.asarray(conditions,dtype=float)
    if ids.ndim!=1 or len(ids)<2 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(values)):
        raise ValueError('distinct valid training indices required')
    scale=np.asarray(stats['scale'],dtype=float)
    threshold=float(options.get('tsr_threshold',2.))
    floor=float(options.get('variance_floor',.25))
    lower=float(options.get('min_weight',.25));upper=float(options.get('max_weight',4.))
    if not np.isfinite([threshold,floor,lower,upper]).all() or floor<=0 or not 0<lower<=upper:
        raise ValueError('invalid loss balancing options')
    train=values[ids]
    if train.ndim!=3 or scale.shape!=(train.shape[-1],) or a.shape!=(len(values),2) or not np.isfinite(train).all() or not np.isfinite(a[ids]).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('invalid training residuals, conditions, or channel scales')
    low=int(np.sum(a[ids,1]<=threshold));high=len(ids)-low
    if not low or not high:raise ValueError('both TSR groups must occur in training data')
    variance=np.mean((train-train.mean(axis=0))**2,axis=0)/scale**2
    clipped=np.clip(1/np.maximum(variance,floor),lower,upper)
    weights=clipped/clipped.mean(axis=0,keepdims=True)
    return dict(fit_indices=ids.tolist(),tsr_threshold=threshold,
                group_counts=dict(low=low,high=high),
                group_weights=dict(low=len(ids)/(2*low),high=len(ids)/(2*high)),
                variance_floor=floor,pre_normalization_clip=[lower,upper],
                angular_weights=weights.tolist(),normalized_local_variance=variance.tolist(),
                angular_weight_min=weights.min(axis=0).tolist(),angular_weight_max=weights.max(axis=0).tolist(),
                scope='Training loss only; train cases only; per-channel mean angular weight is one.')


def sample_loss_weights(conditions,indices,report):
    tsr=np.asarray(conditions)[indices,1]
    return np.where(tsr<=report['tsr_threshold'],report['group_weights']['low'],
                    report['group_weights']['high']).astype('float32')
