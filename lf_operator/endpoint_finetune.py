"""MFFM-inspired one-step endpoint fine-tuning of a pretrained residual operator.

This single-level adaptation preserves the frozen LF and source assets. Its
operating point is deterministic: state=0, time=0.5, one velocity evaluation.
It does not train or evaluate the original stochastic ODE sampling interface.
"""
from pathlib import Path
import json
import numpy as np
import torch


def midpoint_residual(model,condition,n_targets=3):
    field=condition['features']
    state=field.new_zeros(field.shape[0],field.shape[-1],n_targets)
    time=field.new_full((field.shape[0],),.5)
    return model(state,time,condition)


def compose_endpoint(residual,lf,stats):
    if 'spectral_whitening' in stats:raise ValueError('endpoint experiment requires original residual coordinates')
    mean=residual.new_tensor(stats['mean_profile']);scale=residual.new_tensor(stats['scale'])
    return lf+mean+residual*scale


def endpoint_relative_l2(prediction,target):
    """Mean per-case, per-channel relative L2 of final physical HF curves."""
    return (torch.linalg.vector_norm(prediction-target,dim=1)/
            torch.linalg.vector_norm(target,dim=1).clamp_min(1e-8)).mean()


def train_endpoint(cfg):
    from . import blade_pipeline as bp
    from .residual_flow import create_velocity_model
    from .conditioning import standardize_conditions,tensor_conditions,prepare_condition
    device=torch.device(bp.setup(cfg));work=bp.ROOT/cfg['work'];assets=work/cfg.get('flow_assets','stage2')
    out=bp.flow_output_directory(cfg);parent_path=bp.ROOT/cfg['endpoint_parent']['checkpoint']
    if out.exists():raise FileExistsError(out)
    parent_hash=bp.digest(parent_path)
    if parent_hash!=cfg['endpoint_parent']['sha256']:raise ValueError('parent checkpoint identity changed')
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    bp.validate_velocity_checkpoint(parent)
    if bp.read_json(assets/'status.json')['status']!='complete':raise ValueError('incomplete source assets')
    provenance=bp.read_json(assets/'provenance.json')
    if not all(bp.digest(Path(p))==h for p,h in provenance.items()):raise ValueError('frozen assets changed')
    if bp.digest(work/'stage1/best.pt')!=parent['lf_checkpoint_sha256']:raise ValueError('LF checkpoint mismatch')
    if parent['input_contract']!=bp.INPUT_CONTRACT or parent['target_contract']!=bp.TARGET_CONTRACT:
        raise ValueError('parent input/target contract mismatch')
    with np.load(assets/'paired_data.npz') as f:arrays={k:f[k] for k in f.files}
    splits=bp.read_json(assets/'splits.json');stats=bp.read_json(assets/'normalization.json')
    source=bp.read_json(assets/'source.json');scalers=parent['scalers']
    bp.validate_flow_assets(arrays,splits,stats,source,split_method=cfg.get('hf_split',{}).get('method','geometry'))
    if splits!=parent['splits'] or stats!=parent['residual_stats'] or source!=parent['source']:
        raise ValueError('parent training split, normalization or source mismatch')
    if scalers['fit_indices']!=splits['train']:raise ValueError('condition scaler training indices mismatch')
    if parent['config']['flow_model']!=cfg['flow_model']:raise ValueError('fine-tuning must keep parent architecture')
    tensors=tensor_conditions(standardize_conditions(arrays,scalers),arrays,device)
    model=create_velocity_model(**parent['model_config']).to(device)
    model.load_state_dict(parent['model'],strict=True)
    if not all(torch.equal(v.cpu(),parent['model'][k].cpu()) for k,v in model.state_dict().items()):
        raise ValueError('parent weights were not restored exactly')
    target=torch.as_tensor(arrays['target'],dtype=torch.float32,device=device)
    lf=torch.as_tensor(arrays['lf_prediction'],dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['flow_lr'],weight_decay=cfg['flow_weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['flow_epochs'])
    stopping=bp.FlowEarlyStopping(cfg['flow_patience'],cfg['flow_min_delta'])
    rng=np.random.default_rng(cfg['seed']);history=[];out.mkdir()
    metadata=dict(kind='mffm_one_step_endpoint_finetune',inference_mode='deterministic_midpoint',
        parent_checkpoint=str(parent_path),parent_sha256=parent_hash,parent_epoch=parent['epoch'],
        source_sampled=False,state=0.,time=.5,velocity_evaluations=1,
        objective='mean per-case per-channel physical HF relative L2',
        adaptation='single LF-to-HF stage, existing FiLM-FNO and frozen LF; not the paper multi-level cascade',
        frozen_lf_sha256=parent['lf_checkpoint_sha256'],config=cfg)
    bp.save_json(out/'experiment.json',metadata)
    print(json.dumps(dict(event='endpoint_finetuning_started',device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,**metadata)),flush=True)

    @torch.no_grad()
    def evaluate(indices):
        model.eval();indices=np.asarray(indices,dtype=int);parts=[];total=0.
        for start in range(0,len(indices),cfg['batch_size']):
            ids=indices[start:start+cfg['batch_size']]
            residual=midpoint_residual(model,prepare_condition(model,tensors,ids))
            prediction=compose_endpoint(residual,lf[ids],stats)
            total+=float(endpoint_relative_l2(prediction,target[ids]))*len(ids)
            parts.append(prediction.cpu().numpy())
        return np.concatenate(parts),total/len(indices)

    def save_best(epoch):
        torch.save(dict(kind=metadata['kind'],inference_mode=metadata['inference_mode'],model=model.state_dict(),
            model_config=model.config,scalers=scalers,source=source,residual_stats=stats,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),config=cfg,epoch=epoch,
            parent_checkpoint_sha256=parent_hash,parent_epoch=parent['epoch'],splits=splits,
            theta=arrays['theta'].tolist(),targets=bp.TARGETS,lf_checkpoint_sha256=parent['lf_checkpoint_sha256'],
            target_contract=bp.TARGET_CONTRACT,input_contract=bp.INPUT_CONTRACT),out/'best.pt')

    _,best=evaluate(splits['val']);best_epoch=0;save_best(0);stopping.observe(0,best)
    initial=dict(epoch=0,val_endpoint_relative_l2=best,phase='pretrained_one_step_before_finetuning')
    history.append(initial);print(json.dumps(initial),flush=True)
    for epoch in range(1,cfg['flow_epochs']+1):
        model.train();losses=[];rate=optimizer.param_groups[0]['lr']
        for _ in range(cfg['updates_per_epoch']):
            ids=rng.choice(splits['train'],cfg['batch_size'],replace=True)
            unique,inverse=np.unique(ids,return_inverse=True)
            condition={k:v[inverse] for k,v in prepare_condition(model,tensors,unique).items()}
            optimizer.zero_grad(set_to_none=True)
            prediction=compose_endpoint(midpoint_residual(model,condition),lf[ids],stats)
            loss=endpoint_relative_l2(prediction,target[ids])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite endpoint loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()))
        scheduler.step()
        row=dict(epoch=epoch,train_endpoint_relative_l2=float(np.mean(losses)),learning_rate=rate,
                 next_learning_rate=optimizer.param_groups[0]['lr'],optimizer_updates=epoch*cfg['updates_per_epoch'])
        if epoch==1 or epoch%cfg['validate_every']==0 or epoch==cfg['flow_epochs']:
            _,score=evaluate(splits['val']);row['val_endpoint_relative_l2']=score
            stopping.observe(epoch,score)
            if score<best:best=score;best_epoch=epoch;save_best(epoch)
        history.append(row);bp.save_json(out/'history.json',history);print(json.dumps(row),flush=True)
        if stopping.should_stop(epoch):break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model']);model.eval()
    report=dict(kind=metadata['kind'],inference_mode=metadata['inference_mode'],best_epoch=best_epoch,
                best_val_endpoint_relative_l2=best,epochs_run=epoch,parent_epoch=parent['epoch'],
                uncertainty='Deterministic point prediction; no generated uncertainty estimate')
    for split in ('train','val','test'):
        ids=np.asarray(splits[split]);mean,score=evaluate(ids)
        result=dict(corrected=bp.metrics(mean,arrays['target'][ids]),endpoint_relative_l2=score,
            standardized_curve_mse=float(np.mean(((mean-arrays['target'][ids])/np.asarray(stats['scale']))**2)))
        for name,mask in [('low_tsr',arrays['conditions'][ids,1]<=2),('high_tsr',arrays['conditions'][ids,1]>2)]:
            if mask.any():result[name]=dict(count=int(mask.sum()),corrected=bp.metrics(mean[mask],arrays['target'][ids[mask]]))
        report[split]=result
        np.savez_compressed(out/f'{split}_predictions.npz',mean=mean,target=arrays['target'][ids],
            lf=arrays['lf_prediction'][ids],case_ids=arrays['case_ids'][ids],theta=arrays['theta'],targets=np.array(bp.TARGETS))
    comparisons={}
    for label,reference in cfg.get('endpoint_comparisons',{}).items():
        baseline=bp.ROOT/reference;old=bp.read_json(baseline/'metrics.json');comparisons[label]={}
        for split in ('val','test'):
            with np.load(baseline/f'{split}_predictions.npz') as a,np.load(out/f'{split}_predictions.npz') as b:
                for key in ('case_ids','theta','target','lf'):np.testing.assert_array_equal(a[key],b[key])
            comparisons[label][split]={name:dict(old=old[split]['corrected'][name]['rmse'],
                new=report[split]['corrected'][name]['rmse'],reduction_percent=100*(1-
                report[split]['corrected'][name]['rmse']/old[split]['corrected'][name]['rmse'])) for name in bp.TARGETS}
    report['comparisons']=comparisons
    if bp.digest(parent_path)!=parent_hash:raise ValueError('parent checkpoint changed during experiment')
    bp.save_json(out/'metrics.json',report);print(json.dumps(report),flush=True)
    return report
