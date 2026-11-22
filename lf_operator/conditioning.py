"""Train-only scaling and conditioning for the three-channel residual model."""
import numpy as np

AXES={"conditions":(0,),"lf_prediction":(0,1),"rotor_features":(0,2),"field_features":(0,2,3)}


def fit_condition_scalers(arrays,train_indices):
    ids=np.asarray(train_indices,dtype=int)
    if ids.ndim!=1 or len(ids)<2 or len(set(ids.tolist()))!=len(ids):
        raise ValueError('distinct training indices required')
    result={'fit_indices':ids.tolist()}
    for name,axes in AXES.items():
        values=np.asarray(arrays[name],dtype=np.float64)[ids]
        if not np.isfinite(values).all():
            raise ValueError('nonfinite conditioning features')
        result[name]={'mean':values.mean(axis=axes,keepdims=True).tolist(),
                      'scale':np.maximum(values.std(axis=axes,keepdims=True),1e-6).tolist()}
    return result


def standardize_conditions(arrays,stats):
    return {name:((np.asarray(arrays[name])-np.asarray(stats[name]['mean']))/
                  np.asarray(stats[name]['scale'])).astype('float32') for name in AXES}


def tensor_conditions(normalized,arrays,device):
    import torch
    tensors={name:torch.as_tensor(value,dtype=torch.float32,device=device) for name,value in normalized.items()}
    for name in ('z','weights'):
        tensors[name]=torch.as_tensor(arrays[name],dtype=torch.float32,device=device)
    tensors['theta']=torch.as_tensor(arrays['theta'],dtype=torch.float32,device=device)[None].expand(len(arrays['case_ids']),-1)
    return tensors


def prepare_condition(model,tensors,indices):
    return model.prepare_condition(*(tensors[key][indices] for key in
        ('lf_prediction','rotor_features','field_features','conditions','z','weights','theta')))
