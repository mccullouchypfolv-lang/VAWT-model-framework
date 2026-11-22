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
    return ((values-np.asarray(stats['mean_profile']))/np.asarray(stats['scale'])).astype('float32')


def decode_residual(values,stats):
    return np.asarray(values)*np.asarray(stats['scale'])+np.asarray(stats['mean_profile'])


def sample_multichannel_source(spec,count,rng):
    from .spectral_source import sample_source
    return np.concatenate([sample_source(channel,count,rng=rng) for channel in spec['channels']],axis=-1).astype('float32')


def validate_flow_assets(arrays,splits,stats,source):
    sets=[set(splits[k]) for k in ('train','val','test')]
    n=len(arrays['case_ids'])
    if len(set(arrays['case_ids'].tolist()))!=n or any(not s for s in sets) or set.union(*sets)!=set(range(n)):
        raise ValueError('invalid case identities/splits')
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('overlapping HF splits')
    geometry=[set(arrays['geometry_ids'][splits[k]].tolist()) for k in ('train','val','test')]
    if any(geometry[i]&geometry[j] for i in range(3) for j in range(i+1,3)):
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
    work=ROOT/cfg['work']; cache=work/'lf_cache'
    if (cache/'manifest.json').exists():raise FileExistsError(cache/'manifest.json')
    source=(ROOT/cfg['lf_source']).resolve(); config=read_json(source/'common_config.json')
    completed={r['case_id'] for r in table(source/'progress.csv') if r['status']=='completed'}
    inputs={r['case_id']:r for r in table(ROOT/cfg['lf_inputs'])}
    records=[dict(case_id=i,geometry_id=inputs[i]['geometry_id'],sample_id=inputs[i]['sample_id'],file=f'cases/{i}.npz') for i in sorted(completed)]
    hf_root=(ROOT/cfg['hf_root']).resolve()
    folders=sorted(p for p in hf_root.iterdir() if p.is_dir() and p.name.startswith('S'))
    by_id={r['case_id']:r for r in records}
    hf_ids=[p.name for p in folders]
    if len(folders)!=49 or not set(hf_ids)<=by_id.keys():raise ValueError('expected 49 completed LF/HF pairs')
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
    converted_lookup={r['case_id']:r for r in converted}
    hf,provenance=[],{}
    for folder in folders:
        item=read_hf_case(folder,config)
        with np.load(cache/converted_lookup[folder.name]['file']) as d:
            lf=dict(case_id=folder.name,conditions=d['a'],theta=d['theta'],reference_conditions=d['reference_conditions'])
        match_cases(lf,item)
        hf.append(item); provenance[folder.name]=item['metadata']
    np.savez_compressed(work/'hf_data.npz',case_ids=np.array(hf_ids),
                        geometry_ids=np.array([by_id[i]['geometry_id'] for i in hf_ids]),
                        conditions=np.stack([h['a'] for h in hf]),theta=hf[0]['theta'],
                        target=np.stack([h['y'] for h in hf]))
    save_json(work/'hf_provenance.json',provenance)
    save_json(cache/'manifest.json',manifest)
    result=dict(status='prepared',lf_cases=len(converted),hf_cases=49,hf_only_smoke=hf_only,targets=TARGETS,input_contract=INPUT_CONTRACT,
                hf_root=str(hf_root),coefficient_contract='See docs/single-blade-design.md; no old CP/CQ conversion')
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
def lf_predict(checkpoint,cache,manifest,indices,device,n_theta=None):
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
        return lf_predict_parameters(checkpoint,queries,device,n_theta=n_theta)
    model=FiLMFNO(**checkpoint['model_config']).to(device).eval(); model.load_state_dict(checkpoint['model'])
    values={k:[] for k in ('lf_prediction','rotor_features','field_features')}
    for batch in loader(cache,manifest,indices,checkpoint['statistics'],64,require_targets=False):
        result=forward(model,to_device(batch,device),features=True)
        prediction=result['prediction']*result['prediction'].new_tensor(checkpoint['statistics']['y_std'])+result['prediction'].new_tensor(checkpoint['statistics']['y_mean'])
        values['lf_prediction'].append(prediction.cpu().numpy())
        for k in ('rotor_features','field_features'):values[k].append(result[k].cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in values.items()}
    raw=[]
    for i in indices:
        with np.load(cache/manifest['records'][i]['file']) as d:raw.append({k:d[k] for k in ('a','z','weights','theta')})
    arrays.update(case_ids=np.array([manifest['records'][i]['case_id'] for i in indices]),
                  conditions=np.stack([d['a'] for d in raw]),z=(np.stack([d['z'] for d in raw])-manifest['config']['rotor_bottom_height'])/manifest['config']['height'],
                  weights=np.stack([d['weights'] for d in raw]),theta=raw[0]['theta'])
    return arrays


@torch.no_grad()
def lf_predict_parameters(checkpoint,parameters,device,n_theta=None,model=None):
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
    x=(x-np.asarray(stats['x_mean'],dtype='float32')[None,None,:,None])/np.asarray(stats['x_std'],dtype='float32')[None,None,:,None]
    normalized_a=(a-np.asarray(stats['a_mean'],dtype='float32'))/np.asarray(stats['a_std'],dtype='float32')
    z=(z-cfg['rotor_bottom_height'])/cfg['height']
    if model is None:
        model=FiLMFNO(**checkpoint['model_config']).to(device).eval();model.load_state_dict(checkpoint['model'])
    data=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in (x,normalized_a,z,theta,weights)]
    result=model(*data,return_features=True)
    prediction=result['prediction']*result['prediction'].new_tensor(stats['y_std'])+result['prediction'].new_tensor(stats['y_mean'])
    return dict(case_ids=np.array(ids),conditions=a,z=z,weights=weights,theta=theta[0],
                lf_prediction=prediction.cpu().numpy(),rotor_features=result['rotor_features'].cpu().numpy(),
                field_features=result['field_features'].cpu().numpy())


def train_lf(cfg,epochs=None):
    from .model import FiLMFNO
    from .training_data import loader,to_device,forward
    device=setup(cfg); work=ROOT/cfg['work']; cache=work/'lf_cache'; out=work/'stage1'
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    manifest=read_json(cache/'manifest.json'); splits=group_split(manifest['records'],seed=cfg['seed'])
    validate_parameter_contract(manifest)
    if manifest.get('target_contract')!=TARGET_CONTRACT:raise ValueError('LF target contract mismatch')
    for record in manifest['records']:
        if digest(cache/record['file'])!=record['cache_hash']:raise ValueError('LF cache hash mismatch')
    stats=lf_statistics(cache,manifest,splits['train']); out.mkdir(exist_ok=True)
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
    splits=group_split(records,seed=cfg['seed'])
    arrays['residual']=arrays['target']-arrays['lf_prediction']
    stats=residual_normalization(arrays['residual'],splits['train'])
    encoded=encode_residual(arrays['residual'],stats)
    out.mkdir(); np.savez_compressed(out/'paired_data.npz',**arrays)
    save_json(out/'normalization.json',stats); save_json(out/'splits.json',splits)
    channels=[]; failures={}
    for j,name in enumerate(TARGETS):
        try:channels.append(fit_source(encoded[splits['train'],:,j:j+1],seed=cfg['seed'],**cfg.get('source_fit',{})))
        except (AdmissibilityError,ValueError) as exc:failures[name]=str(exc)
    if failures:
        save_json(out/'status.json',dict(status='source_infeasible',failures=failures,targets=TARGETS))
        raise ValueError(f'No admissible source for new targets: {failures}')
    save_json(out/'source.json',dict(channels=channels,targets=TARGETS,target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT,fit_indices=splits['train'],cross_channel_covariance='independent source channels; joint OT'))
    inputs=[work/'stage1'/'best.pt',cache/'manifest.json',work/'hf_data.npz',work/'hf_provenance.json']+[out/n for n in ('paired_data.npz','splits.json','normalization.json','source.json')]
    save_json(out/'provenance.json',{str(p.resolve()):digest(p) for p in inputs})
    lf_train_geometry={manifest['records'][i]['geometry_id'] for i in checkpoint['splits']['train']}
    overlap={k:sorted(set(arrays['geometry_ids'][splits[k]].tolist())&lf_train_geometry) for k in ('val','test')}
    report=dict(status='complete',targets=TARGETS,splits={k:len(v) for k,v in splits.items()},
                beta=[s['beta'] for s in channels],nu=[s['nu'] for s in channels],old_cp_cq_assets_used=False,
                lf_pretraining_seen_hf_holdout_geometries=overlap,
                evaluation_scope='HF labels held out; LF may include the same geometries')
    save_json(out/'status.json',report); print(json.dumps(report),flush=True); return report


def paired_batch(arrays,splits,stats,source,count,rng,pairing):
    from .kernel_ot import couple_endpoints
    indices=rng.choice(splits['train'],count,replace=True)
    g0=sample_multichannel_source(source,count,rng)
    target=encode_residual(arrays['residual'][indices],stats)
    coupled=couple_endpoints(g0,target,rng=rng,**pairing)
    i,j=coupled['source_indices'],coupled['target_indices']
    g0,target,indices=g0[i],target[j],indices[j]
    t=rng.uniform(0,1,(count,1,1)).astype('float32')
    return indices,(1-t)*g0+t*target,t,target-g0,coupled['diagnostics']


@torch.no_grad()
def draw_predictions(model,tensors,arrays,indices,source,stats,count,steps,seed,reference_theta=None):
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
    encoded=integrate_velocity(model,torch.as_tensor(g0,device=next(model.parameters()).device),condition,steps=steps)
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
    if checkpoint.get('model_config',{}).get('architecture')!='floral-filmfno-v1':
        raise ValueError('Velocity architecture changed; retrain stage3 with FiLM-FNO. Existing stage1/stage2 assets can be reused.')


def train_flow(cfg,epochs=None):
    from .conditioning import fit_condition_scalers,standardize_conditions,tensor_conditions,prepare_condition
    from .residual_flow import ConditionalVelocity
    from .objectives import velocity_loss
    device=setup(cfg); work=ROOT/cfg['work']; assets=work/'stage2'; out=flow_output_directory(cfg)
    if out.exists():raise FileExistsError(out)
    if read_json(assets/'status.json')['status']!='complete':raise ValueError('Stage-two source is not admissible/complete; inspect stage2/status.json')
    provenance=read_json(assets/'provenance.json')
    if not all(digest(Path(p))==h for p,h in provenance.items()):raise ValueError('frozen stage-two inputs changed')
    with np.load(assets/'paired_data.npz') as d:arrays={k:d[k] for k in d.files}
    splits=read_json(assets/'splits.json'); stats=read_json(assets/'normalization.json'); source=read_json(assets/'source.json')
    validate_flow_assets(arrays,splits,stats,source)
    scalers=fit_condition_scalers(arrays,splits['train'])
    tensors=tensor_conditions(standardize_conditions(arrays,scalers),arrays,device)
    model=ConditionalVelocity(**cfg['flow_model'],n_targets=3,lf_channels=3,n_conditions=2).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=cfg['flow_lr'],weight_decay=.001)
    rng=np.random.default_rng(cfg['seed']); out.mkdir(); history=[]; best=float('inf'); best_epoch=0
    for epoch in range(1,(epochs or cfg['flow_epochs'])+1):
        model.train(); losses=[]; errors=[]
        for _ in range(cfg['updates_per_epoch']):
            ids,xt,t,u,diagnostics=paired_batch(arrays,splits,stats,source,cfg['batch_size'],rng,cfg['pairing'])
            unique,inverse=np.unique(ids,return_inverse=True)
            condition={k:v[inverse] for k,v in prepare_condition(model,tensors,unique).items()}
            optim.zero_grad(set_to_none=True)
            loss=velocity_loss(model(torch.as_tensor(xt,device=device),torch.as_tensor(t,device=device),condition),torch.as_tensor(u,device=device))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite velocity loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optim.step()
            losses.append(float(loss.detach())); errors.append(diagnostics['marginal_error'])
        row=dict(epoch=epoch,velocity_mse=float(np.mean(losses)),sinkhorn_marginal_error=max(errors))
        if epoch==1 or epoch%cfg['validate_every']==0 or epoch==(epochs or cfg['flow_epochs']):
            model.eval(); result=draw_predictions(model,tensors,arrays,splits['val'],source,stats,cfg['validation_draws'],cfg['ode_steps'],cfg['seed']+10000)
            score=float(np.mean(((result['mean']-arrays['target'][splits['val']])/np.array(stats['scale']))**2))
            row['val_standardized_curve_mse']=score
            if score<best:
                best=score; best_epoch=epoch
                torch.save(dict(model=model.state_dict(),model_config=model.config,scalers=scalers,source=source,residual_stats=stats,
                                config=cfg,targets=TARGETS,theta=arrays['theta'].tolist(),epoch=epoch,splits=splits,
                                lf_checkpoint_sha256=digest(work/'stage1'/'best.pt'),target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT),out/'best.pt')
        history.append(row); save_json(out/'history.json',history); print(json.dumps(row),flush=True)
        if epoch-best_epoch>=cfg['flow_patience']:break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(checkpoint['model']); model.eval()
    report={}
    for split in ('val','test'):
        ids=splits[split]; result=draw_predictions(model,tensors,arrays,ids,source,stats,cfg['evaluation_draws'],cfg['ode_steps'],cfg['seed']+(20000 if split=='test' else 10000))
        mean_baseline=arrays['lf_prediction'][ids]+np.asarray(stats['mean_profile'])
        report[split]=dict(corrected=metrics(result['mean'],arrays['target'][ids]),lf=metrics(arrays['lf_prediction'][ids],arrays['target'][ids]),
                           train_mean=metrics(mean_baseline,arrays['target'][ids]))
        np.savez_compressed(out/f'{split}_predictions.npz',**result,target=arrays['target'][ids],lf=arrays['lf_prediction'][ids],theta=arrays['theta'],case_ids=arrays['case_ids'][ids],targets=np.array(TARGETS))
    report.update(best_epoch=best_epoch,epochs_run=epoch,uncertainty='Uncalibrated generated dispersion',conditional_source_independence_guaranteed=False)
    save_json(out/'metrics.json',report); print(json.dumps(report),flush=True); return report


def predict(cfg,cases,output,parameters=None,lf_only=False,n_theta=None):
    from .conditioning import standardize_conditions,tensor_conditions
    from .residual_flow import ConditionalVelocity
    device=setup(cfg); work=ROOT/cfg['work']; cache=work/'lf_cache'
    out=Path(output)
    if out.exists() or out.suffix!='.npz':raise ValueError('output must be a new .npz path')
    lf_checkpoint=torch.load(work/'stage1'/'best.pt',map_location='cpu',weights_only=False)
    if parameters is not None:arrays=lf_predict_parameters(lf_checkpoint,parameters,device,n_theta=n_theta)
    else:
        manifest=read_json(cache/'manifest.json'); lookup={r['case_id']:i for i,r in enumerate(manifest['records'])}
        arrays=lf_predict(lf_checkpoint,cache,manifest,[lookup[i] for i in cases],device,n_theta=n_theta)
    if lf_only:
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,prediction=arrays['lf_prediction'],theta=arrays['theta'],case_ids=arrays['case_ids'],targets=np.array(TARGETS),input_contract=np.array(INPUT_CONTRACT))
        return {'output':str(out.resolve()),'targets':TARGETS,'input_contract':INPUT_CONTRACT,'model':'LF'}
    ck=torch.load(flow_output_directory(cfg)/'best.pt',map_location='cpu',weights_only=False)
    validate_velocity_checkpoint(ck)
    if ck.get('input_contract')!=INPUT_CONTRACT or ck['targets']!=TARGETS or ck['target_contract']!=TARGET_CONTRACT or digest(work/'stage1'/'best.pt')!=ck['lf_checkpoint_sha256']:raise ValueError('input/target or LF checkpoint mismatch')
    model=ConditionalVelocity(**ck['model_config']).to(device).eval(); model.load_state_dict(ck['model'])
    tensors=tensor_conditions(standardize_conditions(arrays,ck['scalers']),arrays,device)
    result=draw_predictions(model,tensors,arrays,np.arange(len(arrays['case_ids'])),ck['source'],ck['residual_stats'],cfg['evaluation_draws'],cfg['ode_steps'],cfg['seed'],reference_theta=np.asarray(ck['theta']))
    out.parent.mkdir(parents=True,exist_ok=True)
    from .resolution import SOURCE_SCOPE
    np.savez_compressed(out,**result,theta=arrays['theta'],case_ids=arrays['case_ids'],targets=np.array(TARGETS),input_contract=np.array(INPUT_CONTRACT),
                        training_theta=np.asarray(ck['theta']),source_scope=np.array(SOURCE_SCOPE))
    return {'output':str(out.resolve()),'targets':TARGETS}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','stage1','stage2','stage3','predict','predict-lf','predict-grid','predict-grid-lf'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs'/'single_blade.json')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--n-theta',type=int,help='uniform periodic angular points for inference; default is checkpoint grid, or grid specification')
    parser.add_argument('--query-batch-size',type=int,help='condition chunk size for grid inference; training batch size is unchanged')
    parser.add_argument('--hf-only',action='store_true',help='49 LF cases only; explicitly a pipeline smoke dataset')
    query=parser.add_mutually_exclusive_group()
    query.add_argument('--cases',nargs='+'); query.add_argument('--parameters',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); cfg=read_json(args.config)
    if args.n_theta is not None and args.stage not in ('predict','predict-lf','predict-grid','predict-grid-lf'):parser.error('--n-theta is for inference only')
    if args.query_batch_size is not None and args.stage not in ('predict-grid','predict-grid-lf'):parser.error('--query-batch-size is for grid inference only')
    if args.stage=='prepare':prepare(cfg,args.hf_only)
    elif args.stage=='stage1':train_lf(cfg,args.epochs)
    elif args.stage=='stage2':prepare_residuals(cfg)
    elif args.stage=='stage3':train_flow(cfg,args.epochs)
    elif args.stage in ('predict-grid','predict-grid-lf'):
        from .parameter_grid import predict_grid
        if not args.parameters or not args.output or args.cases:parser.error('grid prediction requires --parameters grid.json and --output')
        print(json.dumps(predict_grid(cfg,read_json(args.parameters),args.output,n_theta=args.n_theta,
                                    lf_only=args.stage=='predict-grid-lf',query_batch_size=8 if args.query_batch_size is None else args.query_batch_size)))
    else:
        if (not args.cases and not args.parameters) or not args.output:parser.error('predict requires --cases or --parameters, and --output')
        print(json.dumps(predict(cfg,args.cases,args.output,read_json(args.parameters) if args.parameters else None,lf_only=args.stage=='predict-lf',n_theta=args.n_theta)))
