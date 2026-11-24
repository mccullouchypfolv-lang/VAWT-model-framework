"""Independent SP-FM-inspired trial for conditional function-space source centers."""
from pathlib import Path
import json
import numpy as np
import torch
from .conditional_mixture import (ConditionalMixtureSource,source_parameters,select_mixture_centers,
                                  expected_geodesic_loss,fixed_noise_variance)


@torch.no_grad()
def draw_mixture_predictions(velocity,base,tensors,arrays,indices,source,stats,count,steps,seed,
                            method='dopri5',atol=1e-5,rtol=1e-5,reference_theta=None):
    from . import blade_pipeline as bp
    from .conditioning import prepare_condition
    from .residual_flow import integrate_velocity
    ids=np.asarray(indices,dtype=int);repeated=np.repeat(ids,count)
    means,probabilities=source_parameters(base,tensors,repeated)
    rng=np.random.default_rng(seed);mix_rng=np.random.default_rng(seed+777777)
    if reference_theta is None:
        noise=bp.sample_multichannel_source(source,len(repeated),rng);prediction_stats=stats
    else:
        from .resolution import evaluate_source,evaluate_periodic_profile
        noise=np.concatenate([evaluate_source(channel,len(repeated),arrays['theta'],rng) for channel in source['channels']],axis=-1).astype('float32')
        prediction_stats=dict(stats,mean_profile=evaluate_periodic_profile(np.asarray(stats['mean_profile']),reference_theta,arrays['theta']))
    chosen,component=select_mixture_centers(means,probabilities,means.new_tensor(mix_rng.random(len(repeated))))
    g0=chosen+torch.as_tensor(noise,dtype=means.dtype,device=means.device)
    condition=prepare_condition(velocity,tensors,ids)
    condition={k:v.repeat_interleave(count,0) for k,v in condition.items()}
    endpoint=integrate_velocity(velocity,g0,condition,steps=steps,method=method,atol=atol,rtol=rtol)
    n=len(arrays['theta'])
    encoded=endpoint.cpu().numpy().reshape(len(ids),count,n,3)
    samples=arrays['lf_prediction'][ids,None]+bp.decode_residual(encoded,prediction_stats)
    if not np.isfinite(samples).all():raise FloatingPointError('nonfinite conditional-source flow samples')
    mean_source=(means*probabilities[:,:,None,None]).sum(1)[::count].cpu().numpy()
    source_curve=arrays['lf_prediction'][ids]+bp.decode_residual(mean_source,prediction_stats)
    return dict(samples=samples.astype('float32'),mean=samples.mean(1).astype('float32'),std=samples.std(1).astype('float32'),
        q05=np.quantile(samples,.05,axis=1).astype('float32'),q95=np.quantile(samples,.95,axis=1).astype('float32'),
        source_prediction=source_curve.astype('float32'),mixture_probabilities=probabilities[::count].cpu().numpy(),
        sampled_components=component.cpu().numpy().reshape(len(ids),count))


def train_mixture_source(cfg):
    from . import blade_pipeline as bp
    from .conditioning import (geometry_from_cache,fit_condition_scalers,standardize_conditions,tensor_conditions,prepare_condition)
    from .residual_flow import create_velocity_model
    from .objectives import velocity_loss
    if cfg['pairing']!={'method':'independent'}:raise ValueError('single-target conditional source requires condition-preserving pairing')
    if any(cfg.get(k) for k in ('flow_low_frequency_fm','flow_complex_spectral','flow_loss_balancing','curve_auxiliary','joint_source','flow_bridge')):
        raise ValueError('mixture-source trial excludes other experimental objectives')
    device=bp.setup(cfg);work=bp.ROOT/cfg['work'];assets=work/cfg.get('flow_assets','stage2');out=bp.flow_output_directory(cfg)
    if out.exists():raise FileExistsError(out)
    if bp.read_json(assets/'status.json')['status']!='complete':raise ValueError('incomplete source assets')
    provenance=bp.read_json(assets/'provenance.json')
    if not all(bp.digest(Path(p))==h for p,h in provenance.items()):raise ValueError('frozen inputs changed')
    with np.load(assets/'paired_data.npz') as f:arrays={k:f[k] for k in f.files}
    splits=bp.read_json(assets/'splits.json');stats=bp.read_json(assets/'normalization.json');source=bp.read_json(assets/'source.json')
    bp.validate_flow_assets(arrays,splits,stats,source,split_method=cfg.get('hf_split',{}).get('method','geometry'))
    if np.asarray(stats['scale']).shape!=(3,) or 'spectral_whitening' in stats:raise ValueError('use unchanged global residual scales')
    if 'geometry' not in arrays:arrays=geometry_from_cache(arrays,work/'lf_cache',bp.read_json(work/'lf_cache/manifest.json'))
    scalers=fit_condition_scalers(arrays,splits['train'],include_geometry=True)
    tensors=tensor_conditions(standardize_conditions(arrays,scalers),arrays,device)
    # Construct the unchanged velocity first: source initialization must not alter its seed.
    velocity=create_velocity_model(**cfg['flow_model'],n_targets=3,lf_channels=3,n_conditions=2).to(device)
    base=ConditionalMixtureSource(**cfg['source_model']).to(device)
    optimizer,scheduler=bp.flow_optimization(velocity.parameters(),cfg)
    options=cfg['source_training']
    for key in ('warmup_epochs','cooldown_epochs'):
        if isinstance(options[key],bool) or not isinstance(options[key],int) or options[key]<0:raise ValueError('invalid source phase length')
    every=options['update_every']
    if isinstance(every,bool) or not isinstance(every,int) or every<1:raise ValueError('positive source update frequency required')
    if options['lr']<=0 or options['weight_decay']<0:raise ValueError('invalid source optimizer settings')
    base_optimizer=torch.optim.AdamW(base.parameters(),lr=options['lr'],weight_decay=options['weight_decay'])
    stopping=bp.FlowEarlyStopping(cfg['flow_patience'],cfg['flow_min_delta'])
    target=torch.as_tensor(bp.encode_residual(arrays['residual'],stats),device=device)
    variance=fixed_noise_variance(source);rng=np.random.default_rng(cfg['seed']);mix_rng=np.random.default_rng(cfg['seed']+777777)
    history=[];best=float('inf');best_epoch=0;total_updates=0;source_updates=0;out.mkdir()
    lf_hash=bp.digest(work/'stage1/best.pt')
    metadata=dict(kind='conditional_mixture_source',inference_mode='conditional_mixture_ode',config=cfg,
        source_covariance='unchanged frozen NAGO; conditional means and weights learned',fit_indices=splits['train'],
        objective_adaptation='analytic same-condition geodesic expectation for a single target; no cross-condition OT, no Gumbel approximation',
        fixed_noise_variance=variance,lf_sha256=lf_hash,
        velocity_parameters=sum(p.numel() for p in velocity.parameters()),source_parameters=sum(p.numel() for p in base.parameters()))
    bp.save_json(out/'experiment.json',metadata);print(json.dumps(dict(event='mixture_training_started',**metadata)),flush=True)
    for epoch in range(1,cfg['flow_epochs']+1):
        velocity.train();base.train();losses=[];raw_losses=[];geo_losses=[];entropies=[];separations=[]
        phase='warmup' if epoch<=options['warmup_epochs'] else ('cooldown' if epoch>cfg['flow_epochs']-options['cooldown_epochs'] else 'alternating')
        rate=optimizer.param_groups[0]['lr']
        for _ in range(cfg['updates_per_epoch']):
            ids=rng.choice(splits['train'],cfg['batch_size'],replace=True);count=len(ids)
            noise=bp.sample_multichannel_source(source,count,rng)
            time=torch.as_tensor(rng.uniform(0,1,(count,1,1)).astype('float32'),device=device)
            with torch.no_grad():
                means,probabilities=source_parameters(base,tensors,ids)
                chosen,_=select_mixture_centers(means,probabilities,means.new_tensor(mix_rng.random(count)))
                g0=chosen+torch.as_tensor(noise,device=device);state=(1-time)*g0+time*target[ids];u=target[ids]-g0
                entropy=-(probabilities*probabilities.clamp_min(1e-12).log()).sum(1).mean()
                separation=(means[:,0]-means[:,1]).square().mean().sqrt() if means.shape[1]>1 else means.new_tensor(0.)
                entropies.append(float(entropy));separations.append(float(separation))
            unique,inverse=np.unique(ids,return_inverse=True)
            condition={k:v[inverse] for k,v in prepare_condition(velocity,tensors,unique).items()}
            optimizer.zero_grad(set_to_none=True)
            prediction=velocity(state,time,condition)
            loss=velocity_loss(prediction,u,time=time,time_weighting=cfg['flow_time_weighting'])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite mixture-source FM loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(velocity.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total_updates+=1;losses.append(float(loss.detach()));raw_losses.append(float(velocity_loss(prediction.detach(),u)))
            if phase=='alternating' and total_updates%every==0:
                base_optimizer.zero_grad(set_to_none=True)
                means,probabilities=source_parameters(base,tensors,ids)
                geodesic=expected_geodesic_loss(means,probabilities,target[ids],variance)
                if not torch.isfinite(geodesic):raise FloatingPointError('nonfinite source geodesic loss')
                geodesic.backward();torch.nn.utils.clip_grad_norm_(base.parameters(),1.,error_if_nonfinite=True);base_optimizer.step()
                source_updates+=1;geo_losses.append(float(geodesic.detach()))
        bp.step_flow_scheduler(scheduler,epoch,cfg.get('flow_scheduler_frequency',1))
        row=dict(epoch=epoch,source_phase=phase,velocity_mse=float(np.mean(raw_losses)),time_weighted_velocity_mse=float(np.mean(losses)),
            source_geodesic_mse=float(np.mean(geo_losses)) if geo_losses else None,
            mixture_entropy=float(np.mean(entropies)),center_separation_rms=float(np.mean(separations)),
            learning_rate=rate,optimizer_updates=total_updates,source_optimizer_updates=source_updates)
        if epoch==1 or epoch%cfg['validate_every']==0 or epoch==cfg['flow_epochs']:
            velocity.eval();base.eval()
            result=draw_mixture_predictions(velocity,base,tensors,arrays,splits['val'],source,stats,cfg['validation_draws'],cfg['ode_steps'],cfg['seed']+10000,**bp.ode_options(cfg))
            score=float(np.mean(((result['mean']-arrays['target'][splits['val']])/np.asarray(stats['scale']))**2))
            row['val_standardized_curve_mse']=score
            stopping.observe(epoch,score)
            if score<best:
                best=score;best_epoch=epoch
                torch.save(dict(kind=metadata['kind'],inference_mode=metadata['inference_mode'],model=velocity.state_dict(),model_config=velocity.config,
                    source_model=base.state_dict(),source_model_config=base.config,scalers=scalers,source=source,residual_stats=stats,
                    optimizer=optimizer.state_dict(),source_optimizer=base_optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    config=cfg,epoch=epoch,splits=splits,theta=arrays['theta'].tolist(),targets=bp.TARGETS,
                    lf_checkpoint_sha256=lf_hash,target_contract=bp.TARGET_CONTRACT,input_contract=bp.INPUT_CONTRACT),out/'best.pt')
        history.append(row);bp.save_json(out/'history.json',history);print(json.dumps(row),flush=True)
        if stopping.should_stop(epoch):break
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False)
    velocity.load_state_dict(ck['model']);base.load_state_dict(ck['source_model']);velocity.eval();base.eval()
    report=dict(best_epoch=best_epoch,epochs_run=epoch,kind=metadata['kind'],inference_mode=metadata['inference_mode'],
        uncertainty='Uncalibrated generated dispersion; mixture modes are not identified physical solution modes')
    for split in ('val','test'):
        ids=splits[split]
        result=draw_mixture_predictions(velocity,base,tensors,arrays,ids,source,stats,cfg['evaluation_draws'],cfg['ode_steps'],cfg['seed']+(20000 if split=='test' else 10000),**bp.ode_options(cfg))
        report[split]=dict(corrected=bp.metrics(result['mean'],arrays['target'][ids]),
            source_only=bp.metrics(result['source_prediction'],arrays['target'][ids]),lf=bp.metrics(arrays['lf_prediction'][ids],arrays['target'][ids]))
        for name,mask in [('low_tsr',arrays['conditions'][ids,1]<=2),('high_tsr',arrays['conditions'][ids,1]>2)]:
            if mask.any():report[split][name]=dict(count=int(mask.sum()),corrected=bp.metrics(result['mean'][mask],arrays['target'][np.asarray(ids)[mask]]))
        np.savez_compressed(out/f'{split}_predictions.npz',**result,target=arrays['target'][ids],lf=arrays['lf_prediction'][ids],
            case_ids=arrays['case_ids'][ids],theta=arrays['theta'],targets=np.array(bp.TARGETS))
    comparisons={}
    for label,folder in cfg.get('source_comparisons',{}).items():
        reference=bp.ROOT/folder;old=bp.read_json(reference/'metrics.json');comparisons[label]={}
        for split in ('val','test'):
            with np.load(reference/f'{split}_predictions.npz') as a,np.load(out/f'{split}_predictions.npz') as b:
                for key in ('case_ids','theta','target','lf'):np.testing.assert_array_equal(a[key],b[key])
            comparisons[label][split]={name:dict(old=old[split]['corrected'][name]['rmse'],new=report[split]['corrected'][name]['rmse'],
                reduction_percent=100*(1-report[split]['corrected'][name]['rmse']/old[split]['corrected'][name]['rmse'])) for name in bp.TARGETS}
    report['comparisons']=comparisons
    if bp.digest(work/'stage1/best.pt')!=lf_hash:raise ValueError('frozen LF changed during training')
    bp.save_json(out/'metrics.json',report);print(json.dumps(report),flush=True)
    return report


def predict_mixture(cfg,parameters,output,n_theta=None):
    from . import blade_pipeline as bp
    from .residual_flow import create_velocity_model
    from .conditioning import standardize_conditions,tensor_conditions
    device=bp.setup(cfg);work=bp.ROOT/cfg['work'];out=Path(output)
    if out.exists() or out.suffix!='.npz':raise ValueError('new .npz output required')
    ck=torch.load(bp.flow_output_directory(cfg)/'best.pt',map_location='cpu',weights_only=False)
    if ck.get('inference_mode')!='conditional_mixture_ode':raise ValueError('conditional mixture checkpoint required')
    if bp.digest(work/'stage1/best.pt')!=ck['lf_checkpoint_sha256']:raise ValueError('LF checkpoint mismatch')
    lf=torch.load(work/'stage1/best.pt',map_location='cpu',weights_only=False)
    arrays=bp.lf_predict_parameters(lf,parameters,device,n_theta=n_theta)
    velocity=create_velocity_model(**ck['model_config']).to(device).eval();velocity.load_state_dict(ck['model'])
    base=ConditionalMixtureSource(**ck['source_model_config']).to(device).eval();base.load_state_dict(ck['source_model'])
    tensors=tensor_conditions(standardize_conditions(arrays,ck['scalers']),arrays,device)
    result=draw_mixture_predictions(velocity,base,tensors,arrays,np.arange(len(arrays['case_ids'])),ck['source'],ck['residual_stats'],
        cfg['evaluation_draws'],cfg['ode_steps'],cfg['seed'],reference_theta=np.asarray(ck['theta']),**bp.ode_options(cfg))
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**result,theta=arrays['theta'],case_ids=arrays['case_ids'],targets=np.array(bp.TARGETS))
    return {'output':str(out.resolve()),'inference_mode':ck['inference_mode']}
