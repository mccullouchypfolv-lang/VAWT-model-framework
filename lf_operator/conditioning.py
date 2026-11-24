"""Train-only scaling and conditioning for the three-channel residual model."""
import numpy as np

AXES={"conditions":(0,),"lf_prediction":(0,1),"rotor_features":(0,2),"field_features":(0,2,3)}


def layer_wind_inputs(arrays,profile):
    """Reconstruct prescribed free-stream wind, never induced/relative velocities.

    Cached z is normalized above blade bottom. Pool wind scale across cases AND
    layers, so per-layer standardization does not erase a fixed shear profile.
    """
    geometry=np.asarray(arrays['geometry'],dtype=np.float64)
    a=np.asarray(arrays['conditions'],dtype=np.float64)
    z=np.asarray(arrays['z'],dtype=np.float64)
    if geometry.shape!=(len(a),20,2) or not np.isfinite(geometry).all() or not np.allclose(geometry,geometry[:,:1],rtol=1e-6,atol=1e-6):
        raise ValueError('scalar geometry path requires uniform chord/thickness across heights')
    if z.shape!=(len(a),20) or a.shape!=(len(a),2) or not np.isfinite(z).all() or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('invalid physical conditions or normalized heights')
    height,bottom,reference,exponent=[float(profile[k]) for k in ('height','rotor_bottom_height','wind_reference_height','shear_exponent')]
    if not np.isfinite([height,bottom,reference,exponent]).all() or height<=0 or reference<=0:
        raise ValueError('invalid prescribed wind profile parameters')
    ground=bottom+height*z
    if np.any(ground<=0):raise ValueError('wind heights must be above ground')
    if profile['wind_profile']=='power_law':wind=a[:,0,None]*(ground/reference)**exponent
    elif profile['wind_profile']=='uniform':wind=np.broadcast_to(a[:,0,None],ground.shape).copy()
    else:raise ValueError('unsupported prescribed wind profile')
    return dict(scalar_geometry=geometry[:,0],layer_wind=wind)


def scalar_geometry_input(arrays):
    geometry=np.asarray(arrays['geometry'],dtype=np.float64)
    if geometry.shape!=(len(arrays['conditions']),20,2) or not np.isfinite(geometry).all() or not np.allclose(geometry,geometry[:,:1],rtol=1e-6,atol=1e-6):
        raise ValueError('scalar geometry requires uniform finite chord/thickness across heights')
    return geometry[:,0]


def fit_condition_scalers(arrays,train_indices,include_geometry=False,*,include_field=True,wind_profile=None,include_scalar_geometry=False):
    ids=np.asarray(train_indices,dtype=int)
    if ids.ndim!=1 or len(ids)<2 or len(set(ids.tolist()))!=len(ids):
        raise ValueError('distinct training indices required')
    result={'fit_indices':ids.tolist()}
    axes_by_name=dict(AXES)
    if not include_field:axes_by_name.pop('field_features')
    if include_geometry:axes_by_name['geometry']=(0,1)
    if include_scalar_geometry:
        arrays=dict(arrays,scalar_geometry=scalar_geometry_input(arrays))
        axes_by_name['scalar_geometry']=(0,)
    if wind_profile is not None:
        arrays=dict(arrays,**layer_wind_inputs(arrays,wind_profile))
        axes_by_name.update(scalar_geometry=(0,),layer_wind=(0,1))
        result['wind_profile']=dict(wind_profile)
    for name,axes in axes_by_name.items():
        values=np.asarray(arrays[name],dtype=np.float64)[ids]
        if not np.isfinite(values).all():
            raise ValueError('nonfinite conditioning features')
        result[name]={'mean':values.mean(axis=axes,keepdims=True).tolist(),
                      'scale':np.maximum(values.std(axis=axes,keepdims=True),1e-6).tolist()}
    return result


def standardize_conditions(arrays,stats):
    names=[key for key in AXES if key in stats]+(['geometry'] if 'geometry' in stats else [])
    if 'wind_profile' in stats:
        arrays=dict(arrays,**layer_wind_inputs(arrays,stats['wind_profile']))
        names+=['scalar_geometry','layer_wind']
    elif 'scalar_geometry' in stats:
        arrays=dict(arrays,scalar_geometry=scalar_geometry_input(arrays))
        names+=['scalar_geometry']
    return {name:((np.asarray(arrays[name])-np.asarray(stats[name]['mean']))/
                  np.asarray(stats[name]['scale'])).astype('float32') for name in names}


def tensor_conditions(normalized,arrays,device,*,include_height=True):
    import torch
    tensors={name:torch.as_tensor(value,dtype=torch.float32,device=device) for name,value in normalized.items()}
    for name in (('z','weights') if include_height else ()):
        tensors[name]=torch.as_tensor(arrays[name],dtype=torch.float32,device=device)
    tensors['theta']=torch.as_tensor(arrays['theta'],dtype=torch.float32,device=device)[None].expand(len(arrays['case_ids']),-1)
    return tensors


def prepare_condition(model,tensors,indices):
    if model.config.get('architecture')=='rotor-only-filmfno-v1':
        return model.prepare_condition(*(tensors[key][indices] for key in ('lf_prediction','rotor_features','conditions','theta')))
    kwargs={}
    if model.config.get('architecture')=='angular-state-film-v1':
        if 'scalar_geometry' not in tensors:raise ValueError('angular state model requires scalar geometry')
        kwargs['scalar_geometry']=tensors['scalar_geometry'][indices]
    if model.config.get('architecture')=='layer-wind-film-path-v1':
        for key in ('scalar_geometry','layer_wind'):
            if key not in tensors:raise ValueError('layer wind model requires '+key)
            kwargs[key]=tensors[key][indices]
    if model.config.get('explicit_physical_conditioning',False) or model.config.get('scalar_geometry_film',False) or model.config.get('architecture')=='physical-film-path-v1':
        if 'geometry' not in tensors:raise ValueError('explicit physical model requires geometry inputs')
        kwargs['geometry']=tensors['geometry'][indices]
    return model.prepare_condition(*(tensors[key][indices] for key in
        ('lf_prediction','rotor_features','field_features','conditions','z','weights','theta')),**kwargs)


def geometry_from_cache(arrays,cache,manifest):
    """Attach declared chord/thickness profiles without changing frozen assets."""
    from pathlib import Path
    from .data import digest
    cache=Path(cache);lookup={r['case_id']:r for r in manifest['records']};profiles=[]
    for i,case in enumerate(arrays['case_ids']):
        record=lookup[str(case)];path=cache/record['file']
        if digest(path)!=record['cache_hash']:raise ValueError('geometry input cache hash mismatch')
        with np.load(path,allow_pickle=False) as d:
            x,a=d['x'],d['a']
            if x.ndim!=3 or x.shape[:2]!=(20,2) or not np.isfinite(x).all() or not np.allclose(x,x[:,:,:1]):
                raise ValueError('geometry inputs must be angle-independent [20,2,N]')
            if not np.allclose(a,arrays['conditions'][i],rtol=1e-6,atol=1e-6):
                raise ValueError('geometry and operating condition case mismatch')
            profiles.append(x[:,:,0])
    return dict(arrays,geometry=np.stack(profiles).astype('float32'))
