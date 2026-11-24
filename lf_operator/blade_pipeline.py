"""Independent single-blade Cl/Cd/CM preparation, LF learning and residual FM."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from .data import read_json,save_json,digest,group_split,table
from .blade_data import TARGET_CONTRACT,INPUT_CONTRACT,FEATURES,CONDITIONS,parameter_inputs

ROOT=Path(__file__).resolve().parents[1]
TARGETS=['Cl_blade','Cd_blade','CM_blade']


def validate_parameter_contract(checkpoint,manifest=None):
    if checkpoint.get('input_contract')!=INPUT_CONTRACT or checkpoint.get('features')!=list(FEATURES) or checkpoint.get('conditions')!=list(CONDITIONS):
        raise ValueError('input contract mismatch: old aerodynamic-input models cannot be used')
    if checkpoint.get('targets')!=TARGETS or checkpoint.get('target_contract')!=TARGET_CONTRACT:
        raise ValueError('target contract mismatch')
    if manifest is not None and (manifest.get('input_contract')!=INPUT_CONTRACT or manifest.get('features')!=list(FEATURES) or manifest.get('conditions')!=list(CONDITIONS)):
        raise ValueError('cache input contract mismatch: prepare parameter-only inputs')


def residual_normalization(values,indices):
    train=np.asarray(values,dtype=float)[indices]
    mean=train.mean(0)
    scale=np.sqrt(np.mean((train-mean)**2,axis=(0,1)))
    if train.ndim!=3 or train.shape[-1]!=3 or not np.isfinite(train).all() or (scale<=1e-10).any():
        raise ValueError('need three nondegenerate finite residual channels')
    return dict(mean_profile=mean.tolist(),scale=scale.tolist(),fit_indices=list(indices),targets=TARGETS)


def encode_residual(values,stats):
    if 'spectral_whitening' in stats:
        from .spectral_whitening import whiten
        return whiten(values,stats)
    return ((values-np.asarray(stats['mean_profile']))/np.asarray(stats['scale'])).astype('float32')


def decode_residual(values,stats):
    if 'spectral_whitening' in stats:
        from .spectral_whitening import recolor
        return recolor(values,stats)
    return np.asarray(values)*np.asarray(stats['scale'])+np.asarray(stats['mean_profile'])


def sample_multichannel_source(spec,count,rng):
    from .spectral_source import sample_source
    return np.concatenate([sample_source(channel,count,rng=rng) for channel in spec['channels']],axis=-1).astype('float32')


def validate_flow_assets(arrays,splits,stats,source,split_method='geometry'):
    sets=[set(splits[k]) for k in ('train','val','test')]
    n=len(arrays['case_ids'])
    if len(set(arrays['case_ids'].tolist()))!=n or any(not s for s in sets) or set.union(*sets)!=set(range(n)):
        raise ValueError('invalid case identities/splits')
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('overlapping HF splits')
    geometry=[set(arrays['geometry_ids'][splits[k]].tolist()) for k in ('train','val','test')]
    if split_method not in ('geometry','latin_hypercube'):
        raise ValueError('unknown HF split method')
    if source.get('split_method','geometry')!=split_method:
        raise ValueError('source and requested HF split policies differ')
    if split_method=='geometry' and any(geometry[i]&geometry[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('HF geometry leakage')
    if stats['targets']!=TARGETS or source['targets']!=TARGETS or source['target_contract']!=TARGET_CONTRACT or source.get('input_contract')!=INPUT_CONTRACT or len(source['channels'])!=3:
        raise ValueError('target/source contract mismatch')
    if stats['fit_indices']!=splits['train'] or source['fit_indices']!=splits['train']:
        raise ValueError('source/statistics split mismatch')
    if not np.allclose(arrays['residual'],arrays['target']-arrays['lf_prediction'],atol=1e-6,rtol=1e-6):
        raise ValueError('residuals do not match the frozen baseline')


def setup(cfg):
    seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    device=cfg['device']
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: choose DRL interpreter')
    return device


def prepare(cfg,hf_only=False):
    from .blade_data import read_lf_case,read_hf_case,match_cases,FEATURES
    from .hf_dataset import collect_hf_folders,build_hf_arrays
    work=ROOT/cfg['work']; cache=work/'lf_cache'
    if (cache/'manifest.json').exists():raise FileExistsError(cache/'manifest.json')
    source=(ROOT/cfg['lf_source']).resolve(); config=read_json(source/'common_config.json')
    completed={r['case_id'] for r in table(source/'progress.csv') if r['status']=='completed'}
    inputs={r['case_id']:r for r in table(ROOT/cfg['lf_inputs'])}
    records=[dict(case_id=i,geometry_id=inputs[i]['geometry_id'],sample_id=inputs[i]['sample_id'],file=f'cases/{i}.npz') for i in sorted(completed)]
    folders=collect_hf_folders(cfg,ROOT)
    by_id={r['case_id']:r for r in records}
    hf_ids=[p.name for p in folders]
    if not set(hf_ids)<=by_id.keys():raise ValueError('HF cases missing completed LF inputs')
    records=[r for r in records if not hf_only or r['case_id'] in set(hf_ids)]
    (cache/'cases').mkdir(parents=True,exist_ok=True)
    def convert(record):
        item=read_lf_case(source/'cases'/record['case_id'],config)
        hashes=item['metadata'].get('source_hashes',{})
        np.savez(cache/record['file'],**{k:np.asarray(item[k],dtype='float32') for k in ('x','a','z','weights','theta','y','baseline','reference_conditions')})
        return dict(record,cache_hash=digest(cache/record['file']),source_hashes=hashes)
    with ThreadPoolExecutor(max_workers=cfg.get('workers',8)) as pool:
        converted=list(pool.map(convert,records))
    manifest=dict(schema='single-blade-parameters-v1',input_contract=INPUT_CONTRACT,target_contract=TARGET_CONTRACT,source=str(source),config=config,features=list(FEATURES),
                  conditions=list(CONDITIONS),targets=TARGETS,records=converted,hf_only_smoke=hf_only,
                  input_policy='Only U_ref, TSR and 20-layer chord/thickness; aerodynamic values are labels only')
    hf,provenance=build_hf_arrays(folders,cache,manifest)
    np.savez_compressed(work/'hf_data.npz',**hf)
    save_json(work/'hf_provenance.json',provenance)
    save_json(cache/'manifest.json',manifest)
    result=dict(status='prepared',lf_cases=len(converted),hf_cases=len(folders),hf_only_smoke=hf_only,targets=TARGETS,input_contract=INPUT_CONTRACT,
                hf_roots=sorted({str(p.parent) for p in folders}),coefficient_contract='See docs/single-blade-design.md; no old CP/CQ conversion')
    save_json(work/'preparation.json',result); print(json.dumps(result),flush=True)
    return result


def lf_statistics(cache,manifest,indices):
    sums={}; squares={}; counts={}
    for i in indices:
        with np.load(cache/manifest['records'][i]['file']) as d:
            values=dict(x=d['x'].transpose(0,2,1).reshape(-1,d['x'].shape[1]),a=d['a'][None],y=d['y'])
            for k,v in values.items():
                v=v.astype(float); sums[k]=sums.get(k,0)+v.sum(0); squares[k]=squares.get(k,0)+(v*v).sum(0)
                counts[k]=counts.get(k,0)+len(v)
    stats={}
    for k in sums:
        mean=sums[k]/counts[k]; scale=np.sqrt(np.maximum(squares[k]/counts[k]-mean*mean,0))
        stats[k+'_mean']=mean.tolist(); stats[k+'_std']=np.maximum(scale,1e-6).tolist()
    stats['fit_indices']=list(indices)
    return stats


def loss_function(prediction,target):
    delta=prediction-target
    spectrum=torch.fft.rfft(delta,dim=1,norm='ortho')
    # Frequency weighting adds information beyond unweighted Parseval-equivalent MSE.
    weights=torch.arange(spectrum.shape[1],device=delta.device,dtype=delta.dtype)/max(1,spectrum.shape[1]-1)
    spectral=(spectrum.abs().square()*weights[None,:,None]).mean()
    derivative=(delta.roll(-1,1)-delta.roll(1,1))/2
    return delta.square().mean()+.01*spectral+.05*derivative.square().mean()


def metrics(prediction,target):
    error=np.asarray(prediction)-np.asarray(target)
    if error.shape[-1]!=3 or not np.isfinite(error).all():raise ValueError('invalid three-target predictions')
    return {name:dict(rmse=float(np.sqrt(np.mean(error[...,i]**2))),mae=float(np.mean(abs(error[...,i])))) for i,name in enumerate(TARGETS)}


@torch.no_grad()
def lf_predict(checkpoint,cache,manifest,indices,device,n_theta=None,*,return_field_features=True):
    from .model import FiLMFNO
    from .training_data import loader,to_device,forward
    validate_parameter_contract(checkpoint,manifest)
    if checkpoint['targets']!=TARGETS or manifest['targets']!=TARGETS or checkpoint['target_contract']!=TARGET_CONTRACT:raise ValueError('old target checkpoint/cache rejected')
    if digest(cache/'manifest.json')!=checkpoint['manifest_sha256']:raise ValueError('LF manifest differs from frozen checkpoint')
    for i in indices:
        if digest(cache/manifest['records'][i]['file'])!=manifest['records'][i]['cache_hash']:raise ValueError('LF cache hash mismatch')
    if n_theta is not None:
        queries=[]
        for i in indices:
            record=manifest['records'][i]
            with np.load(cache/record['file']) as data:
                x,a=data['x'],data['a']
                queries.append(dict(case_id=record['case_id'],wind_ref_m_s=float(a[0]),tsr_ref=float(a[1]),
                                    chord_m=x[:,0,0],thickness_ratio=x[:,1,0]))
        return lf_predict_parameters(checkpoint,queries,device,n_theta=n_theta,return_field_features=return_field_features)
    model=FiLMFNO(**checkpoint['model_config']).to(device).eval(); model.load_state_dict(checkpoint['model'])
    feature_keys=('rotor_features','field_features') if return_field_features else ('rotor_features',)
    values={k:[] for k in ('lf_prediction',*feature_keys)}
    for batch in loader(cache,manifest,indices,checkpoint['statistics'],64,require_targets=False):
        result=forward(model,to_device(batch,device),features=True)
        prediction=result['prediction']*result['prediction'].new_tensor(checkpoint['statistics']['y_std'])+result['prediction'].new_tensor(checkpoint['statistics']['y_mean'])
        values['lf_prediction'].append(prediction.cpu().numpy())
        for k in feature_keys:values[k].append(result[k].cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in values.items()}
    raw=[]
    for i in indices:
        with np.load(cache/manifest['records'][i]['file']) as d:raw.append({k:d[k] for k in ('a','z','weights','theta','x')})
    arrays.update(case_ids=np.array([manifest['records'][i]['case_id'] for i in indices]),
                  conditions=np.stack([d['a'] for d in raw]),z=(np.stack([d['z'] for d in raw])-manifest['config']['rotor_bottom_height'])/manifest['config']['height'],
                  weights=np.stack([d['weights'] for d in raw]),theta=raw[0]['theta'],
                  geometry=np.stack([d['x'][:,:,0] for d in raw]))
    return arrays


@torch.no_grad()
def lf_predict_parameters(checkpoint,parameters,device,n_theta=None,model=None,*,return_field_features=True):
    """Inference from declared variables only. No LF/HF solver outputs are opened."""
    from .model import FiLMFNO
    validate_parameter_contract(checkpoint)
    input_config=dict(checkpoint['source_config'])
    if n_theta is not None:input_config['ntheta']=n_theta
    if isinstance(parameters,dict):parameters=[parameters]
    if not parameters:raise ValueError('at least one parameter query required')
    inputs=[]; ids=[]
    required={'wind_ref_m_s','tsr_ref','chord_m','thickness_ratio'}
    for i,query in enumerate(parameters):
        if not required<=query.keys() or set(query)-required-{'case_id'}:
            raise ValueError('query must contain only U_ref, TSR, chord and thickness profiles, plus optional case_id')
        inputs.append(parameter_inputs(query['wind_ref_m_s'],query['tsr_ref'],query['chord_m'],query['thickness_ratio'],input_config))
        ids.append(str(query.get('case_id',f'query_{i:04d}')))
    stats=checkpoint['statistics']; cfg=checkpoint['source_config']
    x=np.stack([v['x'] for v in inputs]).astype('float32'); a=np.stack([v['a'] for v in inputs]).astype('float32')
    z=np.stack([v['z'] for v in inputs]).astype('float32'); weights=np.stack([v['weights'] for v in inputs]).astype('float32')
    theta=np.stack([v['theta'] for v in inputs]).astype('float32')
    geometry=x[:,:,:,0].copy()
    x=(x-np.asarray(stats['x_mean'],dtype='float32')[None,None,:,None])/np.asarray(stats['x_std'],dtype='float32')[None,None,:,None]
    normalized_a=(a-np.asarray(stats['a_mean'],dtype='float32'))/np.asarray(stats['a_std'],dtype='float32')
    z=(z-cfg['rotor_bottom_height'])/cfg['height']
    if model is None:
        model=FiLMFNO(**checkpoint['model_config']).to(device).eval();model.load_state_dict(checkpoint['model'])
    data=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in (x,normalized_a,z,theta,weights)]
    result=model(*data,return_features=True)
    prediction=result['prediction']*result['prediction'].new_tensor(stats['y_std'])+result['prediction'].new_tensor(stats['y_mean'])
    arrays=dict(case_ids=np.array(ids),conditions=a,z=z,weights=weights,theta=theta[0],
                lf_prediction=prediction.cpu().numpy(),rotor_features=result['rotor_features'].cpu().numpy(),
                geometry=geometry)
    if return_field_features:arrays['field_features']=result['field_features'].cpu().numpy()
    return arrays


def train_lf(cfg,epochs=None):
    from .model import FiLMFNO
    from .training_data import loader,to_device,forward
    device=setup(cfg); work=ROOT/cfg['work']; cache=work/'lf_cache'; out=work/'stage1'
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    manifest=read_json(cache/'manifest.json')
    validate_parameter_contract(manifest)
    if manifest.get('target_contract')!=TARGET_CONTRACT:raise ValueError('LF target contract mismatch')
    for record in manifest['records']:
        if digest(cache/record['file'])!=record['cache_hash']:raise ValueError('LF cache hash mismatch')
    split_cfg=cfg.get('lf_split',{});split_method=split_cfg.get('method','geometry')
    if split_method=='geometry':
        splits=group_split(manifest['records'],seed=cfg['seed'])
        split_report=dict(method=split_method,seed=cfg['seed'])
    elif split_method=='latin_hypercube':
        from .splitting import case_parameter_inputs,latin_hypercube_split
        parameters=case_parameter_inputs(cache,manifest['records'])
        splits,split_report=latin_hypercube_split(parameters,seed=cfg['seed'],
            val_fraction=split_cfg.get('val_fraction',.15),test_fraction=split_cfg.get('test_fraction',.15))
        split_report['dimensions']=['wind_ref_m_s','tsr_ref','chord_m','thickness_ratio']
    else:raise ValueError('unknown LF split method')
    stats=lf_statistics(cache,manifest,splits['train']); out.mkdir(exist_ok=True)
    save_json(out/'splits.json',splits);save_json(out/'split_design.json',split_report)
    model_cfg=dict(cfg['lf_model'],n_features=2,n_conditions=2,n_outputs=3)
    model=FiLMFNO(**model_cfg).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=cfg['lf_lr'],weight_decay=.0001)
    history=[]; best=float('inf'); best_epoch=0
    for epoch in range(1,(epochs or cfg['lf_epochs'])+1):
        model.train(); total=0.; count=0
        for batch in loader(cache,manifest,splits['train'],stats,cfg['batch_size'],shuffle=True):
            b=to_device(batch,device); optim.zero_grad(set_to_none=True)
            loss=loss_function(forward(model,b),b['y']); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optim.step()
            total+=float(loss.detach())*len(b['x']); count+=len(b['x'])
        model.eval(); valid=0.; nv=0
        with torch.no_grad():
            for batch in loader(cache,manifest,splits['val'],stats,cfg['batch_size']):
                b=to_device(batch,device); valid+=float(loss_function(forward(model,b),b['y']))*len(b['x']); nv+=len(b['x'])
        row=dict(epoch=epoch,train_loss=total/count,val_loss=valid/nv); history.append(row); print(json.dumps(row),flush=True)
        if row['val_loss']<best:
            best=row['val_loss']; best_epoch=epoch
            torch.save(dict(model=model.state_dict(),model_config=model.config,statistics=stats,splits=splits,targets=TARGETS,
                            config=cfg,epoch=epoch,manifest_sha256=digest(cache/'manifest.json'),target_contract=TARGET_CONTRACT,
                            input_contract=INPUT_CONTRACT,features=list(FEATURES),conditions=list(CONDITIONS),source_config=manifest['config']),out/'best.pt')
        save_json(out/'history.json',history)
        if epoch-best_epoch>=cfg['lf_patience']:break
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    report={}
    for split in ('val','test'):
        prediction=lf_predict(checkpoint,cache,manifest,splits[split],device)['lf_prediction']
        target=np.stack([np.load(cache/manifest['records'][i]['file'])['y'] for i in splits[split]])
        report[split]=metrics(prediction,target)
    save_json(out/'metrics.json',report); return report


def prepare_residuals(cfg):
    if cfg.get('source_design',{}).get('method')=='floral_fixed_matern':
        from .floral_source import prepare_floral_assets
        return prepare_floral_assets(cfg)
    from .spectral_source import fit_source,AdmissibilityError
    device=setup(cfg); work=ROOT/cfg['work']; out=work/'stage2'
    if out.exists():raise FileExistsError(out)
    cache=work/'lf_cache'; manifest=read_json(cache/'manifest.json')
    checkpoint=torch.load(work/'stage1'/'best.pt',map_location='cpu',weights_only=False)
    if digest(cache/'manifest.json')!=checkpoint['manifest_sha256']:raise ValueError('LF manifest differs from checkpoint')
    with np.load(work/'hf_data.npz') as d:hf={k:d[k] for k in d.files}
    lookup={r['case_id']:i for i,r in enumerate(manifest['records'])}
    arrays=lf_predict(checkpoint,cache,manifest,[lookup[i] for i in hf['case_ids']],device)
    if not np.allclose(arrays['theta'],hf['theta'],atol=1e-6,rtol=0):raise ValueError('HF/LF phase mismatch')
    arrays.update(target=hf['target'],geometry_ids=hf['geometry_ids'])
    records=[dict(case_id=str(i),geometry_id=str(g)) for i,g in zip(hf['case_ids'],hf['geometry_ids'])]
    split_cfg=cfg.get('hf_split',{});split_method=split_cfg.get('method','geometry')
    split_report=dict(method=split_method,seed=cfg['seed'])
    if split_method=='geometry':
        splits=group_split(records,seed=cfg['seed'])
    elif split_method=='latin_hypercube':
        from .splitting import latin_hypercube_split
        from .splitting import case_parameter_inputs
        parameters=case_parameter_inputs(cache,[manifest['records'][lookup[case]] for case in hf['case_ids']])
        splits,split_report=latin_hypercube_split(parameters,seed=cfg['seed'],
            val_fraction=split_cfg.get('val_fraction',.15),test_fraction=split_cfg.get('test_fraction',.15))
        split_report['dimensions']=['wind_ref_m_s','tsr_ref','chord_m','thickness_ratio']
    else:raise ValueError('unknown HF split method')
    arrays['residual']=arrays['target']-arrays['lf_prediction']
    stats=residual_normalization(arrays['residual'],splits['train'])
    encoded=encode_residual(arrays['residual'],stats)
    out.mkdir(); np.savez_compressed(out/'paired_data.npz',**arrays)
    save_json(out/'normalization.json',stats); save_json(out/'splits.json',splits)
    save_json(out/'split_design.json',split_report)
    channels=[]; failures={}
    for j,name in enumerate(TARGETS):
        try:channels.append(fit_source(encoded[splits['train'],:,j:j+1],seed=cfg['seed'],**cfg.get('source_fit',{})))
        except (AdmissibilityError,ValueError) as exc:failures[name]=str(exc)
    if failures:
        save_json(out/'status.json',dict(status='source_infeasible',failures=failures,targets=TARGETS))
        raise ValueError(f'No admissible source for new targets: {failures}')
    save_json(out/'source.json',dict(channels=channels,targets=TARGETS,target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT,fit_indices=splits['train'],split_method=split_method,cross_channel_covariance='independent source channels; joint OT'))
    inputs=[work/'stage1'/'best.pt',cache/'manifest.json',work/'hf_data.npz',work/'hf_provenance.json']+[out/n for n in ('paired_data.npz','splits.json','normalization.json','source.json','split_design.json')]
    save_json(out/'provenance.json',{str(p.resolve()):digest(p) for p in inputs})
    lf_train_geometry={manifest['records'][i]['geometry_id'] for i in checkpoint['splits']['train']}
    overlap={k:sorted(set(arrays['geometry_ids'][splits[k]].tolist())&lf_train_geometry) for k in ('val','test')}
    report=dict(status='complete',targets=TARGETS,splits={k:len(v) for k,v in splits.items()},
                beta=[s['beta'] for s in channels],nu=[s['nu'] for s in channels],old_cp_cq_assets_used=False,
                lf_pretraining_seen_hf_holdout_geometries=overlap,
                split_method=split_method,
                evaluation_scope=('HF labels held out; LF may include the same geometries' if split_method=='geometry'
                                  else 'HF cases held out; geometry overlap allowed; approximate LHS input coverage'))
    save_json(out/'status.json',report); print(json.dumps(report),flush=True); return report


def coupling_method(pairing,joint_enabled=False):
    method=pairing.get('method','kernel_ot')
    if method not in ('kernel_ot','independent'):raise ValueError('unknown coupling method')
    if method=='independent' and (set(pairing)!={'method'} or joint_enabled):
        raise ValueError('independent coupling must not contain kernel options or enable joint source/coupling optimization')
    return method


def paired_batch(arrays,splits,stats,source,count,rng,pairing,joint_pairing=None,*,target_indices=None,bridge=None):
    method=coupling_method(pairing,joint_enabled=joint_pairing is not None)
    if target_indices is None:
        indices=rng.choice(splits['train'],count,replace=True)
    else:
        indices=np.asarray(target_indices,dtype=np.int64)
        if indices.shape!=(count,) or not set(indices.tolist())<=set(splits['train']):raise ValueError('invalid training batch indices')
    g0=sample_multichannel_source(source,count,rng)
    target=encode_residual(arrays['residual'][indices],stats)
    if method=='independent':
        # Source draws are independent of the target/condition draws. No
        # finite batch transport plan or endpoint-dependent reordering.
        diagnostics=dict(method='independent',conditional_source_independence_by_construction=True)
    else:
        options={k:v for k,v in pairing.items() if k!='method'}
        if joint_pairing is None:
            from .kernel_ot import couple_endpoints
            coupled=couple_endpoints(g0,target,rng=rng,**options)
        else:
            from .joint_source_coupling import couple_joint_endpoints
            coupled=couple_joint_endpoints(g0,target,arrays['coupling_condition_ids'][indices],rng=rng,**options,**joint_pairing)
        i,j=coupled['source_indices'],coupled['target_indices']
        g0,target,indices=g0[i],target[j],indices[j]
        diagnostics=coupled['diagnostics']
    t=rng.uniform(0,1,(count,1,1)).astype('float32')
    if bridge is None:state=(1-t)*g0+t*target
    else:
        from .flow_training import flow_path
        state=flow_path(g0,target,t,source,rng,bridge)
    return indices,state,t,target-g0,diagnostics


def ode_options(cfg):
    return dict(method=cfg.get('ode_method','heun'),atol=cfg.get('ode_atol',1e-5),rtol=cfg.get('ode_rtol',1e-5))


@torch.no_grad()
def draw_predictions(model,tensors,arrays,indices,source,stats,count,steps,seed,reference_theta=None,
                     method='heun',atol=1e-5,rtol=1e-5,solver_stats=None):
    from .conditioning import prepare_condition
    from .residual_flow import integrate_velocity
    condition=prepare_condition(model,tensors,indices)
    condition={k:v.repeat_interleave(count,0) for k,v in condition.items()}
    rng=np.random.default_rng(seed)
    if reference_theta is None:
        g0=sample_multichannel_source(source,len(indices)*count,rng)
        prediction_stats=stats
    else:
        from .resolution import evaluate_source,evaluate_periodic_profile
        if any(channel['n_theta']!=len(reference_theta) for channel in source['channels']):
            raise ValueError('source and training reference grid mismatch')
        g0=np.concatenate([evaluate_source(channel,len(indices)*count,arrays['theta'],rng)
                           for channel in source['channels']],axis=-1).astype('float32')
        prediction_stats=dict(stats,mean_profile=evaluate_periodic_profile(
            np.asarray(stats['mean_profile']),reference_theta,arrays['theta']))
        if np.asarray(stats['scale']).ndim==2:
            prediction_stats['scale']=np.exp(evaluate_periodic_profile(
                np.log(np.asarray(stats['scale'])),reference_theta,arrays['theta']))
    encoded=integrate_velocity(model,torch.as_tensor(g0,device=next(model.parameters()).device),condition,
                               steps=steps,method=method,atol=atol,rtol=rtol,diagnostics=solver_stats)
    residual=decode_residual(encoded.cpu().numpy().reshape(len(indices),count,len(arrays['theta']),3),prediction_stats)
    samples=arrays['lf_prediction'][indices,None]+residual
    if not np.isfinite(samples).all():raise FloatingPointError('nonfinite flow samples')
    return dict(samples=samples.astype('float32'),mean=samples.mean(1).astype('float32'),std=samples.std(1).astype('float32'),
                q05=np.quantile(samples,.05,axis=1).astype('float32'),q95=np.quantile(samples,.95,axis=1).astype('float32'))


def flow_output_directory(cfg):
    """Select a separate velocity run without relocating frozen LF/source assets."""
    name=cfg.get('flow_output','stage3')
    if (not isinstance(name,str) or not name or name in ('.','..','stage1','stage2','lf_cache')
            or '/' in name or '\\' in name or ':' in name):
        raise ValueError('flow_output must be a distinct stage-three directory name')
    return ROOT/cfg['work']/name


def validate_velocity_checkpoint(checkpoint):
    if checkpoint.get('inference_mode','ode')!='ode':
        raise ValueError(f"{checkpoint['inference_mode']} checkpoint requires its dedicated inference interface, not generic ODE sampling")
    if checkpoint.get('model_config',{}).get('architecture') not in ('floral-filmfno-v1','floral-filmfno-aligned-v2','raw-physical-film-v1','target-decoupled-film-v1','rotor-only-filmfno-v1','physical-film-path-v1','layer-wind-film-path-v1','angular-state-film-v1'):
        raise ValueError('Velocity architecture changed; retrain stage3 with FiLM-FNO. Existing stage1/stage2 assets can be reused.')


class FlowEarlyStopping:
    """Track significant validation improvement; patience is measured in epochs."""
    def __init__(self,patience,min_delta=0.):
        if isinstance(patience,bool) or not isinstance(patience,int) or patience<1:
            raise ValueError('flow patience must be a positive integer')
        if not np.isfinite(min_delta) or min_delta<0:
            raise ValueError('flow minimum improvement must be finite and nonnegative')
        self.patience=patience; self.min_delta=float(min_delta)
        self.reference=float('inf'); self.last_improvement=None

    def observe(self,epoch,score):
        if not np.isfinite(score):raise FloatingPointError('nonfinite validation score')
        if score<self.reference-self.min_delta:
            self.reference=score; self.last_improvement=epoch

    def should_stop(self,epoch):
        return self.last_improvement is not None and epoch-self.last_improvement>=self.patience


def flow_optimization(parameters,cfg):
    """Keep historical configs reproducible; new runs opt into epoch decay."""
    lr=cfg['flow_lr']; decay=cfg.get('flow_weight_decay',.001); gamma=cfg.get('flow_lr_gamma',1.)
    for name,value in [('flow_lr',lr),('flow_weight_decay',decay),('flow_lr_gamma',gamma)]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    if lr<=0 or decay<0 or not 0<gamma<=1:
        raise ValueError('flow_lr must be positive, weight decay nonnegative and gamma in (0,1]')
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=decay)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)
    return optimizer,scheduler


def step_flow_scheduler(scheduler,epoch,frequency=1):
    """Equivalent to an epoch scheduler with Lightning frequency=N."""
    if isinstance(frequency,bool) or not isinstance(frequency,int) or frequency<1:
        raise ValueError('flow scheduler frequency must be a positive integer')
    if epoch%frequency==0:scheduler.step()


@torch.no_grad()
def validation_velocity_mse(model,tensors,arrays,indices,source,stats,*,draws,batch_size,seed,device,
                            time_weighting='none',bridge=None,low_frequency=None,complex_spectral=None,return_components=False):
    """Evaluate independent FM paths on every held-out condition with fixed draws."""
    from .conditioning import prepare_condition
    from .objectives import velocity_loss,complex_spectral_loss
    from .flow_training import flow_path
    for name,value in [('draws',draws),('batch_size',batch_size)]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(name+' must be a positive integer')
    indices=np.asarray(indices,dtype=np.int64)
    if indices.ndim!=1 or not len(indices):raise ValueError('validation indices required')
    rng=np.random.default_rng(seed);ids=np.tile(indices,draws);total=0.;base_total=0.;spectral_total=0.;complex_total=0.
    for start in range(0,len(ids),batch_size):
        chosen=ids[start:start+batch_size];count=len(chosen)
        g0=sample_multichannel_source(source,count,rng)
        target=encode_residual(arrays['residual'][chosen],stats)
        time=rng.uniform(0,1,(count,1,1)).astype('float32')
        state=flow_path(g0,target,time,source,rng,bridge)
        prediction=model(torch.as_tensor(state,device=device),torch.as_tensor(time,device=device),
                         prepare_condition(model,tensors,chosen))
        loss=velocity_loss(prediction,torch.as_tensor(target-g0,device=device),
                           time=torch.as_tensor(time,device=device),time_weighting=time_weighting)
        base_total+=float(loss)*count
        if low_frequency is not None:
            from .low_frequency_fm import low_frequency_velocity_loss
            spectral=low_frequency_velocity_loss(prediction,torch.as_tensor(target-g0,device=device),
                low_frequency['weights'],time=torch.as_tensor(time,device=device),time_weighting=time_weighting)
            spectral_total+=float(spectral)*count
            loss=loss+low_frequency['strength']*spectral
        if complex_spectral is not None:
            complex_loss=complex_spectral_loss(prediction,torch.as_tensor(target-g0,device=device),
                time=torch.as_tensor(time,device=device),time_weighting=time_weighting)
            complex_total+=float(complex_loss)*count
            loss=loss+complex_spectral['strength']*complex_loss
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite validation velocity loss')
        total+=float(loss)*count
    if return_components:
        return dict(total=total/len(ids),fm_mse=base_total/len(ids),low_frequency_fm_mse=spectral_total/len(ids),
                    complex_spectral_l1=complex_total/len(ids))
    return total/len(ids)


def curve_endpoint_loss(model,tensors,arrays,splits,source,stats,rng,*,cases=4,draws=2,steps=16,method='heun'):
    """Differentiate the integrated normalized residual mean, using train cases only.

    This is a true ODE rollout, not x_t+(1-t)*v(x_t). The normalization makes
    endpoint residual MSE equal to channel-balanced corrected-HF curve MSE.
    """
    from .conditioning import prepare_condition
    from .residual_flow import integrate_velocity
    for name,value in [('cases',cases),('draws',draws),('steps',steps)]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'curve auxiliary {name} must be a positive integer')
    if method not in ('euler','heun'):raise ValueError('curve auxiliary requires euler or heun')
    if not splits['train']:raise ValueError('curve auxiliary requires training cases')
    ids=rng.choice(splits['train'],min(cases,len(splits['train'])),replace=False)
    condition={k:v.repeat_interleave(draws,0) for k,v in prepare_condition(model,tensors,ids).items()}
    parameter=next(model.parameters())
    g0=torch.as_tensor(sample_multichannel_source(source,len(ids)*draws,rng),device=parameter.device,dtype=parameter.dtype)
    endpoint=integrate_velocity(model,g0,condition,steps=steps,method=method)
    prediction=endpoint.reshape(len(ids),draws,endpoint.shape[1],endpoint.shape[2]).mean(1)
    target=torch.as_tensor(encode_residual(arrays['residual'][ids],stats),device=parameter.device,dtype=parameter.dtype)
    return (prediction-target).square().mean(),ids


def train_flow(cfg,epochs=None):
    from .conditioning import fit_condition_scalers,standardize_conditions,tensor_conditions,prepare_condition
    from .residual_flow import create_velocity_model as ConditionalVelocity
    from .objectives import velocity_loss,validate_time_weighting,complex_spectral_loss,validate_velocity_regression
    from .flow_training import epoch_indices,validate_bridge
    method=coupling_method(cfg['pairing'],joint_enabled=cfg.get('joint_source',{}).get('enabled',False))
    rotor_only=cfg['flow_model'].get('architecture')=='rotor-only-filmfno-v1'
    time_weighting=validate_time_weighting(cfg.get('flow_time_weighting','none'))
    regression=validate_velocity_regression(cfg.get('flow_velocity_loss','mse'),cfg.get('flow_huber_delta',1.))
    regression_options=dict(regression=regression,huber_delta=cfg.get('flow_huber_delta',1.))
    epoch_mode=cfg.get('flow_epoch_mode','monte_carlo')
    validation_metric=cfg.get('flow_validation_metric','ode_curve_mse')
    if validation_metric not in ('ode_curve_mse','velocity_mse'):raise ValueError('unknown flow validation metric')
    scheduler_frequency=cfg.get('flow_scheduler_frequency',1)
    if isinstance(scheduler_frequency,bool) or not isinstance(scheduler_frequency,int) or scheduler_frequency<1:
        raise ValueError('flow scheduler frequency must be a positive integer')
    complex_spectral=cfg.get('flow_complex_spectral')
    if complex_spectral is not None:
        if not isinstance(complex_spectral,dict) or set(complex_spectral)!={'strength'}:
            raise ValueError('complex spectral options must contain strength only')
        strength=complex_spectral['strength']
        if isinstance(strength,bool) or not isinstance(strength,(float,int)) or not np.isfinite(strength) or strength<0:
            raise ValueError('complex spectral strength must be finite and nonnegative')
    if epoch_mode not in ('monte_carlo','dataloader'):raise ValueError('unknown flow epoch mode')
    if epoch_mode=='dataloader' and method!='independent':raise ValueError('full-pass batches require independent coupling to preserve target coverage')
    bridge=cfg.get('flow_bridge');validate_bridge(bridge)
    device=setup(cfg); work=ROOT/cfg['work']; assets=work/cfg.get('flow_assets','stage2'); out=flow_output_directory(cfg)
    if out.exists():raise FileExistsError(out)
    if read_json(assets/'status.json')['status']!='complete':raise ValueError('Stage-two source is not admissible/complete; inspect stage2/status.json')
    provenance=read_json(assets/'provenance.json')
    if not all(digest(Path(p))==h for p,h in provenance.items()):raise ValueError('frozen stage-two inputs changed')
    with np.load(assets/'paired_data.npz') as d:
        arrays={k:d[k] for k in d.files if not rotor_only or k not in ('field_features','z','weights')}
    splits=read_json(assets/'splits.json'); stats=read_json(assets/'normalization.json'); source=read_json(assets/'source.json')
    validate_flow_assets(arrays,splits,stats,source,split_method=cfg.get('hf_split',{}).get('method','geometry'))
    loss_balance=None
    if cfg.get('flow_loss_balancing'):
        from .loss_balancing import fit_loss_weights,sample_loss_weights
        loss_balance=fit_loss_weights(arrays['residual'],arrays['conditions'],splits['train'],stats,cfg['flow_loss_balancing'])
    joint_pairing=None
    if cfg.get('joint_source',{}).get('enabled',False):
        from .joint_source_coupling import load_joint_source
        source,arrays,joint_pairing=load_joint_source(cfg,arrays)
        validate_flow_assets(arrays,splits,stats,source,split_method=cfg.get('hf_split',{}).get('method','geometry'))
    low_frequency=None
    if cfg.get('flow_low_frequency_fm') is not None:
        from .low_frequency_fm import fit_low_frequency_weights,low_frequency_velocity_loss
        low_frequency=fit_low_frequency_weights(arrays['residual'],splits['train'],stats,cfg['flow_low_frequency_fm'])
    explicit=cfg['flow_model'].get('explicit_physical_conditioning',False) or cfg['flow_model'].get('scalar_geometry_film',False) or cfg['flow_model'].get('architecture')=='physical-film-path-v1'
    layer_wind=cfg['flow_model'].get('architecture')=='layer-wind-film-path-v1'
    angular_state=cfg['flow_model'].get('architecture')=='angular-state-film-v1'
    wind_profile=None
    if explicit or layer_wind or angular_state:
        from .conditioning import geometry_from_cache
        manifest=read_json(work/'lf_cache/manifest.json')
        arrays=geometry_from_cache(arrays,work/'lf_cache',manifest)
        if layer_wind:
            wind_profile={key:manifest['config'][key] for key in ('wind_profile','height','rotor_bottom_height','wind_reference_height','shear_exponent')}
    scalers=fit_condition_scalers(arrays,splits['train'],include_geometry=explicit,include_field=not rotor_only,wind_profile=wind_profile,include_scalar_geometry=angular_state)
    tensors=tensor_conditions(standardize_conditions(arrays,scalers),arrays,device,include_height=not rotor_only)
    model=ConditionalVelocity(**cfg['flow_model'],n_targets=3,lf_channels=3,n_conditions=2).to(device)
    optim,scheduler=flow_optimization(model.parameters(),cfg)
    stopping=FlowEarlyStopping(cfg['flow_patience'],cfg.get('flow_min_delta',0.))
    curve_cfg=cfg.get('curve_auxiliary',{})
    curve_weight=curve_cfg.get('weight',0.)
    if isinstance(curve_weight,bool) or not isinstance(curve_weight,(int,float)) or not np.isfinite(curve_weight) or curve_weight<0:
        raise ValueError('curve auxiliary weight must be finite and nonnegative')
    curve_options={k:v for k,v in curve_cfg.items() if k!='weight'}
    curve_rng=np.random.default_rng(cfg['seed']+50000)
    rng=np.random.default_rng(cfg['seed']); out.mkdir(); history=[]; best=float('inf'); best_epoch=0; total_updates=0
    if loss_balance is not None:save_json(out/'loss_weights.json',loss_balance)
    if low_frequency is not None:save_json(out/'low_frequency_weights.json',low_frequency)
    for epoch in range(1,(epochs or cfg['flow_epochs'])+1):
        model.train(); losses=[]; weighted_losses=[]; errors=[]; curve_losses=[]; total_losses=[]; joint_diagnostics=[]; learning_rate=optim.param_groups[0]['lr']
        batch_sizes=[];visited=[];spectral_losses=[];complex_losses=[]
        batches=epoch_indices(splits['train'],batch_size=cfg['batch_size'],seed=cfg['seed'],epoch=epoch) if epoch_mode=='dataloader' else (None for _ in range(cfg['updates_per_epoch']))
        for batch_indices in batches:
            count=cfg['batch_size'] if batch_indices is None else len(batch_indices)
            ids,xt,t,u,diagnostics=paired_batch(arrays,splits,stats,source,count,rng,cfg['pairing'],joint_pairing=joint_pairing,target_indices=batch_indices,bridge=bridge)
            batch_sizes.append(count);visited.extend(ids.tolist())
            if joint_pairing is not None:joint_diagnostics.append(diagnostics)
            unique,inverse=np.unique(ids,return_inverse=True)
            condition={k:v[inverse] for k,v in prepare_condition(model,tensors,unique).items()}
            optim.zero_grad(set_to_none=True)
            time=torch.as_tensor(t,device=device);target_velocity=torch.as_tensor(u,device=device)
            prediction=model(torch.as_tensor(xt,device=device),time,condition)
            loss_options={}
            if loss_balance is not None:
                loss_options=dict(sample_weights=sample_loss_weights(arrays['conditions'],ids,loss_balance),
                                  angular_weights=loss_balance['angular_weights'])
            base_loss=velocity_loss(prediction,target_velocity,time=time,time_weighting=time_weighting,**loss_options,**regression_options)
            loss=base_loss
            if low_frequency is not None:
                spectral=low_frequency_velocity_loss(prediction,target_velocity,low_frequency['weights'],
                    time=time,time_weighting=time_weighting)
                loss=base_loss+low_frequency['strength']*spectral
                spectral_losses.append(float(spectral.detach()))
            if complex_spectral is not None:
                complex_loss=complex_spectral_loss(prediction,target_velocity,time=time,time_weighting=time_weighting)
                loss=loss+complex_spectral['strength']*complex_loss
                complex_losses.append(float(complex_loss.detach()))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite velocity loss')
            loss.backward()
            total_value=float(loss.detach())
            if curve_weight:
                # Separate forward/backward releases the large FM graph before
                # constructing the rollout graph; parameters stay fixed until step.
                curve_loss,_=curve_endpoint_loss(model,tensors,arrays,splits,source,stats,curve_rng,**curve_options)
                if not torch.isfinite(curve_loss):raise FloatingPointError('nonfinite curve auxiliary loss')
                (curve_weight*curve_loss).backward()
                curve_losses.append(float(curve_loss.detach()))
                total_value+=curve_weight*curve_losses[-1]
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optim.step()
            total_updates+=1
            losses.append(float(velocity_loss(prediction.detach(),target_velocity)) if time_weighting!='none' or loss_balance is not None or regression!='mse' else float(base_loss.detach()))
            weighted_losses.append(float(base_loss.detach()))
            if 'marginal_error' in diagnostics:errors.append(diagnostics['marginal_error'])
            total_losses.append(total_value)
        step_flow_scheduler(scheduler,epoch,scheduler_frequency)
        row=dict(epoch=epoch,velocity_mse=float(np.average(losses,weights=batch_sizes)),
                 learning_rate=learning_rate,next_learning_rate=optim.param_groups[0]['lr'])
        if loss_balance is not None:
            row['balanced_velocity_loss']=float(np.average(weighted_losses,weights=batch_sizes))
        if epoch_mode=='dataloader':
            if sorted(visited)!=sorted(splits['train']):raise RuntimeError('training epoch did not visit each condition exactly once')
            row.update(epoch_mode=epoch_mode,epoch_samples=sum(batch_sizes),epoch_unique_conditions=len(set(visited)),
                       updates_in_epoch=len(batch_sizes),optimizer_updates=total_updates)
        if bridge is not None:row['bridge_method']=bridge['method']
        if errors:row['sinkhorn_marginal_error']=max(errors)
        if method=='independent':row['coupling_method']='independent'
        if time_weighting!='none':
            row.update(time_weighting=time_weighting)
            row['time_weighted_velocity_'+regression]=float(np.average(weighted_losses,weights=batch_sizes))
        if regression!='mse':
            row.update(velocity_objective=regression,huber_delta=regression_options['huber_delta'],
                       optimized_velocity_loss=float(np.average(total_losses,weights=batch_sizes)))
        if low_frequency is not None:
            row.update(low_frequency_fm_loss=float(np.average(spectral_losses,weights=batch_sizes)),
                       optimized_velocity_loss=float(np.average(total_losses,weights=batch_sizes)))
        if complex_spectral is not None:
            row.update(complex_spectral_l1=float(np.average(complex_losses,weights=batch_sizes)),
                       optimized_velocity_loss=float(np.average(total_losses,weights=batch_sizes)))
        if joint_diagnostics:
            row.update({k:float(np.mean([d[k] for d in joint_diagnostics])) for k in
                ('independent_fraction','condition_chi2_before','condition_chi2_after','kernel_transport_cost','transport_kl','joint_objective')})
        if curve_weight:
            row.update(ode_curve_mse=float(np.mean(curve_losses)),weighted_curve_loss=curve_weight*float(np.mean(curve_losses)),
                       total_loss=float(np.mean(total_losses)))
        if epoch==1 or epoch%cfg['validate_every']==0 or epoch==(epochs or cfg['flow_epochs']):
            model.eval()
            if validation_metric=='velocity_mse':
                score=validation_velocity_mse(model,tensors,arrays,splits['val'],source,stats,
                    draws=cfg['validation_draws'],batch_size=cfg['batch_size'],seed=cfg['seed']+10000,
                    device=device,time_weighting=time_weighting,bridge=bridge,low_frequency=low_frequency,
                    complex_spectral=complex_spectral,return_components=low_frequency is not None or complex_spectral is not None)
                key='val_time_weighted_velocity_mse' if time_weighting!='none' else 'val_velocity_mse'
                if low_frequency is not None or complex_spectral is not None:
                    row[key]=score['fm_mse']
                    if low_frequency is not None:row['val_low_frequency_fm_loss']=score['low_frequency_fm_mse']
                    if complex_spectral is not None:row['val_complex_spectral_l1']=score['complex_spectral_l1']
                    row['val_optimized_velocity_loss']=score['total'];score=score['total']
                else:row[key]=score
            else:
                result=draw_predictions(model,tensors,arrays,splits['val'],source,stats,cfg['validation_draws'],cfg['ode_steps'],cfg['seed']+10000,**ode_options(cfg))
                score=float(np.mean(((result['mean']-arrays['target'][splits['val']])/np.array(stats.get('validation_scale',stats['scale'])))**2))
                row['val_standardized_curve_mse']=score
            stopping.observe(epoch,score)
            if score<best:
                best=score; best_epoch=epoch
                torch.save(dict(model=model.state_dict(),model_config=model.config,scalers=scalers,source=source,residual_stats=stats,
                                optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),
                                config=cfg,targets=TARGETS,theta=arrays['theta'].tolist(),epoch=epoch,splits=splits,
                                loss_balancing=loss_balance,**({'low_frequency_fm':low_frequency} if low_frequency is not None else {}),
                                lf_checkpoint_sha256=digest(work/'stage1'/'best.pt'),target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT),out/'best.pt')
        history.append(row); save_json(out/'history.json',history); print(json.dumps(row),flush=True)
        if stopping.should_stop(epoch):break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(checkpoint['model']); model.eval()
    report={}
    for split in ('val','test'):
        ids=splits[split]; result=draw_predictions(model,tensors,arrays,ids,source,stats,cfg['evaluation_draws'],cfg['ode_steps'],cfg['seed']+(20000 if split=='test' else 10000),**ode_options(cfg))
        mean_baseline=arrays['lf_prediction'][ids]+np.asarray(stats['mean_profile'])
        report[split]=dict(corrected=metrics(result['mean'],arrays['target'][ids]),lf=metrics(arrays['lf_prediction'][ids],arrays['target'][ids]),
                           train_mean=metrics(mean_baseline,arrays['target'][ids]))
        np.savez_compressed(out/f'{split}_predictions.npz',**result,target=arrays['target'][ids],lf=arrays['lf_prediction'][ids],theta=arrays['theta'],case_ids=arrays['case_ids'][ids],targets=np.array(TARGETS))
    report.update(best_epoch=best_epoch,epochs_run=epoch,uncertainty='Uncalibrated generated dispersion',
                  coupling_method=('joint_kernel_ot' if joint_pairing is not None else method),
                  conditional_source_independence_guaranteed=method=='independent')
    save_json(out/'metrics.json',report); print(json.dumps(report),flush=True); return report


def predict(cfg,cases,output,parameters=None,lf_only=False,n_theta=None):
    from .conditioning import standardize_conditions,tensor_conditions
    from .residual_flow import create_velocity_model as ConditionalVelocity
    device=setup(cfg); work=ROOT/cfg['work']; cache=work/'lf_cache'
    out=Path(output)
    if out.exists() or out.suffix!='.npz':raise ValueError('output must be a new .npz path')
    lf_checkpoint=torch.load(work/'stage1'/'best.pt',map_location='cpu',weights_only=False)
    rotor_only=not lf_only and cfg.get('flow_model',{}).get('architecture')=='rotor-only-filmfno-v1'
    if parameters is not None:arrays=lf_predict_parameters(lf_checkpoint,parameters,device,n_theta=n_theta,return_field_features=not rotor_only)
    else:
        manifest=read_json(cache/'manifest.json'); lookup={r['case_id']:i for i,r in enumerate(manifest['records'])}
        arrays=lf_predict(lf_checkpoint,cache,manifest,[lookup[i] for i in cases],device,n_theta=n_theta,return_field_features=not rotor_only)
    if lf_only:
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,prediction=arrays['lf_prediction'],theta=arrays['theta'],case_ids=arrays['case_ids'],targets=np.array(TARGETS),input_contract=np.array(INPUT_CONTRACT))
        return {'output':str(out.resolve()),'targets':TARGETS,'input_contract':INPUT_CONTRACT,'model':'LF'}
    ck=torch.load(flow_output_directory(cfg)/'best.pt',map_location='cpu',weights_only=False)
    validate_velocity_checkpoint(ck)
    if ck.get('input_contract')!=INPUT_CONTRACT or ck['targets']!=TARGETS or ck['target_contract']!=TARGET_CONTRACT or digest(work/'stage1'/'best.pt')!=ck['lf_checkpoint_sha256']:raise ValueError('input/target or LF checkpoint mismatch')
    model=ConditionalVelocity(**ck['model_config']).to(device).eval(); model.load_state_dict(ck['model'])
    tensors=tensor_conditions(standardize_conditions(arrays,ck['scalers']),arrays,device,include_height=not rotor_only)
    result=draw_predictions(model,tensors,arrays,np.arange(len(arrays['case_ids'])),ck['source'],ck['residual_stats'],cfg['evaluation_draws'],cfg['ode_steps'],cfg['seed'],reference_theta=np.asarray(ck['theta']),**ode_options(cfg))
    out.parent.mkdir(parents=True,exist_ok=True)
    from .resolution import source_scope
    np.savez_compressed(out,**result,theta=arrays['theta'],case_ids=arrays['case_ids'],targets=np.array(TARGETS),input_contract=np.array(INPUT_CONTRACT),
                        training_theta=np.asarray(ck['theta']),source_scope=np.array(source_scope(ck['source'])))
    return {'output':str(out.resolve()),'targets':TARGETS}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','prepare-hf','stage1','stage2','stage2-v2','stage3','predict','predict-lf','predict-grid','predict-grid-lf'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs'/'single_blade.json')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--n-theta',type=int,help='uniform periodic angular points for inference; default is checkpoint grid, or grid specification')
    parser.add_argument('--query-batch-size',type=int,help='condition chunk size for grid inference; training batch size is unchanged')
    parser.add_argument('--hf-only',action='store_true',help='paired LF cases only; explicitly a pipeline smoke dataset')
    query=parser.add_mutually_exclusive_group()
    query.add_argument('--cases',nargs='+'); query.add_argument('--parameters',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); cfg=read_json(args.config)
    if args.n_theta is not None and args.stage not in ('predict','predict-lf','predict-grid','predict-grid-lf'):parser.error('--n-theta is for inference only')
    if args.query_batch_size is not None and args.stage not in ('predict-grid','predict-grid-lf'):parser.error('--query-batch-size is for grid inference only')
    if args.stage=='prepare':prepare(cfg,args.hf_only)
    elif args.stage=='prepare-hf':
        from .hf_dataset import prepare_hf
        prepare_hf(cfg)
    elif args.stage=='stage1':train_lf(cfg,args.epochs)
    elif args.stage=='stage2':prepare_residuals(cfg)
    elif args.stage=='stage2-v2':
        from .joint_source_coupling import prepare_joint_source
        prepare_joint_source(cfg)
    elif args.stage=='stage3':train_flow(cfg,args.epochs)
    elif args.stage in ('predict-grid','predict-grid-lf'):
        from .parameter_grid import predict_grid
        if not args.parameters or not args.output or args.cases:parser.error('grid prediction requires --parameters grid.json and --output')
        print(json.dumps(predict_grid(cfg,read_json(args.parameters),args.output,n_theta=args.n_theta,
                                    lf_only=args.stage=='predict-grid-lf',query_batch_size=8 if args.query_batch_size is None else args.query_batch_size)))
    else:
        if (not args.cases and not args.parameters) or not args.output:parser.error('predict requires --cases or --parameters, and --output')
        print(json.dumps(predict(cfg,args.cases,args.output,read_json(args.parameters) if args.parameters else None,lf_only=args.stage=='predict-lf',n_theta=args.n_theta)))
