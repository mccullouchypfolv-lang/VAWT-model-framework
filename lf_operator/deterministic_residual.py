"""Condition-only residual regression control using the unchanged FiLM-FNO."""
import numpy as np
import torch

from .residual_flow import ConditionalVelocity


class ConditionalResidualMean(ConditionalVelocity):
    """Fix unused state/time inputs; regress the endpoint without source or ODE."""
    def forward(self,condition):
        features=condition['features']
        batch,_,nt=features.shape
        state=features.new_zeros(batch,nt,self.config['n_targets'])
        time=features.new_zeros(batch)
        return super().forward(state,time,condition)


def train_deterministic(cfg):
    from pathlib import Path
    from .blade_pipeline import (ROOT,setup,read_json,save_json,digest,validate_flow_assets,
        flow_output_directory,flow_optimization,FlowEarlyStopping,encode_residual,
        decode_residual,metrics,TARGETS,TARGET_CONTRACT,INPUT_CONTRACT)
    from .conditioning import fit_condition_scalers,standardize_conditions,tensor_conditions,prepare_condition
    import json

    if cfg.get('flow_loss_balancing') or cfg.get('curve_auxiliary') or cfg.get('joint_source',{}).get('enabled'):
        raise ValueError('deterministic control excludes additional experiment objectives')
    device=torch.device(setup(cfg));work=ROOT/cfg['work'];assets=work/'stage2';out=flow_output_directory(cfg)
    if out.exists():raise FileExistsError(out)
    if read_json(assets/'status.json')['status']!='complete':raise ValueError('incomplete stage2 assets')
    provenance=read_json(assets/'provenance.json')
    if not all(digest(Path(p))==h for p,h in provenance.items()):raise ValueError('frozen inputs changed')
    with np.load(assets/'paired_data.npz') as f:arrays={k:f[k] for k in f.files}
    splits=read_json(assets/'splits.json');stats=read_json(assets/'normalization.json')
    validate_flow_assets(arrays,splits,stats,read_json(assets/'source.json'),split_method='geometry')
    if cfg['flow_model'].get('explicit_physical_conditioning') or cfg['flow_model'].get('channel_mlp_skip'):
        raise ValueError('control requires original condition path and original blocks')
    scalers=fit_condition_scalers(arrays,splits['train'])
    tensors=tensor_conditions(standardize_conditions(arrays,scalers),arrays,device)
    model=ConditionalResidualMean(**cfg['flow_model'],n_targets=3,lf_channels=3,n_conditions=2).to(device)
    optim,scheduler=flow_optimization(model.parameters(),cfg)
    stopping=FlowEarlyStopping(cfg['flow_patience'],cfg['flow_min_delta'])
    targets=torch.as_tensor(encode_residual(arrays['residual'],stats),device=device)
    rng=np.random.default_rng(cfg['seed']);history=[];best=float('inf');best_epoch=0
    out.mkdir()
    save_json(out/'experiment.json',dict(kind='deterministic_residual_mean',config=cfg,
        state_time='fixed zero',source_sampling=False,ode=False,device=str(device),
        gpu_name=torch.cuda.get_device_name() if device.type=='cuda' else None,
        fit_indices=splits['train'],stage2_provenance=provenance))
    print(json.dumps(dict(event='training_started',device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,config=cfg),ensure_ascii=False),flush=True)

    @torch.no_grad()
    def evaluate(ids):
        ids=np.asarray(ids,dtype=int);parts=[]
        for start in range(0,len(ids),64):
            jj=ids[start:start+64];parts.append(model(prepare_condition(model,tensors,jj)).cpu().numpy())
        encoded=np.concatenate(parts)
        mean=arrays['lf_prediction'][ids]+decode_residual(encoded,stats)
        score=float(np.mean(((mean-arrays['target'][ids])/np.asarray(stats['scale']))**2))
        return mean,score

    for epoch in range(1,cfg['flow_epochs']+1):
        model.train();losses=[]
        for _ in range(cfg['updates_per_epoch']):
            ids=rng.choice(splits['train'],cfg['batch_size'],replace=True)
            unique,inverse=np.unique(ids,return_inverse=True)
            cond={k:v[inverse] for k,v in prepare_condition(model,tensors,unique).items()}
            optim.zero_grad(set_to_none=True)
            loss=(model(cond)-targets[ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite deterministic loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optim.step();losses.append(float(loss.detach()))
        scheduler.step();row=dict(epoch=epoch,train_standardized_residual_mse=float(np.mean(losses)),
                                  learning_rate=optim.param_groups[0]['lr'])
        if epoch==1 or epoch%cfg['validate_every']==0 or epoch==cfg['flow_epochs']:
            model.eval();_,score=evaluate(splits['val']);row['val_standardized_curve_mse']=score
            stopping.observe(epoch,score)
            if score<best:
                best=score;best_epoch=epoch
                torch.save(dict(kind='deterministic_residual_mean',model=model.state_dict(),model_config=model.config,
                    scalers=scalers,residual_stats=stats,config=cfg,epoch=epoch,splits=splits,targets=TARGETS,
                    theta=arrays['theta'].tolist(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),
                    lf_checkpoint_sha256=digest(work/'stage1/best.pt'),target_contract=TARGET_CONTRACT,
                    input_contract=INPUT_CONTRACT),out/'best.pt')
        history.append(row);save_json(out/'history.json',history);print(json.dumps(row),flush=True)
        if stopping.should_stop(epoch):break
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model']);model.eval()
    report=dict(kind='deterministic_residual_mean',best_epoch=best_epoch,epochs_run=epoch)
    for split in ['train','val','test']:
        ids=np.asarray(splits[split]);mean,score=evaluate(ids)
        result=dict(corrected=metrics(mean,arrays['target'][ids]),standardized_curve_mse=score)
        for label,mask in [('low_tsr',arrays['conditions'][ids,1]<=2),('high_tsr',arrays['conditions'][ids,1]>2)]:
            if mask.any():result[label]=dict(count=int(mask.sum()),corrected=metrics(mean[mask],arrays['target'][ids[mask]]))
        report[split]=result
        np.savez_compressed(out/f'{split}_predictions.npz',mean=mean,target=arrays['target'][ids],
            lf=arrays['lf_prediction'][ids],case_ids=arrays['case_ids'][ids],theta=arrays['theta'],targets=np.array(TARGETS))
    baseline=work/'stage3_nago_independent_timeweighted_modes6_e2000'
    with np.load(baseline/'test_predictions.npz') as f:
        lookup={str(c):i for i,c in enumerate(arrays['case_ids'])};ids=np.array([lookup[str(c)] for c in f['case_ids']])
        if ids.tolist()!=list(splits['test']):raise ValueError('test comparison ordering mismatch')
        np.testing.assert_array_equal(f['target'],arrays['target'][ids])
        np.testing.assert_array_equal(f['lf'],arrays['lf_prediction'][ids])
        old_mean=f['mean'].copy()
    report['comparison_original_flow']={}
    mean,_=evaluate(ids)
    for label,mask in [('all',np.ones(len(ids),dtype=bool)),('low_tsr',arrays['conditions'][ids,1]<=2),('high_tsr',arrays['conditions'][ids,1]>2)]:
        old_metrics=metrics(old_mean[mask],arrays['target'][ids[mask]]);new_metrics=metrics(mean[mask],arrays['target'][ids[mask]])
        report['comparison_original_flow'][label]={name:dict(old=old_metrics[name]['rmse'],new=new_metrics[name]['rmse'],
            reduction_percent=100*(1-new_metrics[name]['rmse']/old_metrics[name]['rmse'])) for name in TARGETS}
    save_json(out/'metrics.json',report);print(json.dumps(report),flush=True)
    return report
