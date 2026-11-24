"""Grouped HF cross-validation with fold-local priors and out-of-fold metrics."""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import numpy as np


def _group_test_folds(groups,strata,indices,k,seed):
    names=np.unique(groups[indices])
    if isinstance(k,bool) or not isinstance(k,int) or k<2 or len(names)<k:
        raise ValueError('need at least k distinct geometry groups')
    counts=np.array([np.bincount(strata[indices[groups[indices]==g]],minlength=3) for g in names])
    sizes=counts.sum(1);totals=np.maximum(counts.sum(0),1);rng=np.random.default_rng(seed)
    order=sorted(rng.permutation(len(names)),key=lambda j:(-np.max(counts[j]/totals),-sizes[j]))
    fold_counts=np.zeros((k,3));fold_sizes=np.zeros(k);assignments={}
    for j in order:
        costs=[]
        for fold in range(k):
            trial=fold_counts.copy();trial[fold]+=counts[j]
            size_trial=fold_sizes.copy();size_trial[fold]+=sizes[j]
            costs.append(float(np.std(trial/totals,axis=0).mean()+.25*np.std(size_trial/len(indices))))
        choices=np.flatnonzero(np.isclose(costs,min(costs),rtol=0,atol=1e-12))
        fold=int(rng.choice(choices));fold_counts[fold]+=counts[j];fold_sizes[fold]+=sizes[j];assignments[names[j]]=fold
    return [np.array([i for i in indices if assignments[groups[i]]==f],dtype=int) for f in range(k)]


def grouped_cv_splits(groups,tsr,*,k=5,seed=42,inner_k=5):
    groups=np.asarray(groups).astype(str);tsr=np.asarray(tsr,dtype=float)
    if groups.ndim!=1 or tsr.shape!=groups.shape or not np.isfinite(tsr).all():raise ValueError('matching finite groups/TSR required')
    strata=np.where(tsr<=2,0,np.where(tsr<=3,1,2));indices=np.arange(len(groups));result=[]
    for fold,test in enumerate(_group_test_folds(groups,strata,indices,k,seed)):
        pool=np.setdiff1d(indices,test)
        inner=_group_test_folds(groups,strata,pool,inner_k,seed+100+fold)
        val=inner[fold%inner_k];train=np.setdiff1d(pool,val)
        if not len(train) or not len(val) or not len(test):raise ValueError('empty nested fold')
        result.append(dict(train=train.tolist(),val=val.tolist(),test=test.tolist()))
    return result


def fit_fold_source(arrays,splits,cfg):
    from .blade_pipeline import residual_normalization,encode_residual,TARGETS,TARGET_CONTRACT,INPUT_CONTRACT
    from .spectral_source import fit_source
    stats=residual_normalization(arrays['residual'],splits['train'])
    encoded=encode_residual(arrays['residual'][splits['train']],stats)
    channels=[fit_source(encoded[:,:,c:c+1],seed=cfg['seed'],**cfg.get('source_fit',{})) for c in range(3)]
    source=dict(channels=channels,targets=TARGETS,target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT,
                fit_indices=splits['train'],split_method='geometry',cross_channel_covariance='independent source channels')
    return stats,source


def error_summary(prediction,target):
    prediction=np.asarray(prediction,dtype=float);target=np.asarray(target,dtype=float)
    if prediction.shape!=target.shape or prediction.ndim!=3 or not len(prediction):raise ValueError('matching nonempty curve tensors required')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():raise ValueError('nonfinite predictions')
    error=prediction-target;per_case=np.sqrt(np.mean(error**2,axis=1))
    return dict(count=len(prediction),pooled_rmse=np.sqrt(np.mean(error**2,axis=(0,1))).tolist(),
        per_case_rmse_median=np.median(per_case,axis=0).tolist(),
        per_case_rmse_q25=np.quantile(per_case,.25,axis=0).tolist(),per_case_rmse_q75=np.quantile(per_case,.75,axis=0).tolist(),
        per_case_rmse_mean=per_case.mean(0).tolist(),per_case_rmse_p90=np.quantile(per_case,.9,axis=0).tolist(),
        per_case_rmse_max=per_case.max(0).tolist())


def assemble_oof(n_cases,fold_predictions,shape):
    prediction=np.empty((n_cases,*shape),dtype=float);assignment=np.full(n_cases,-1,dtype=int)
    for fold,ids,values in fold_predictions:
        ids=np.asarray(ids,dtype=int);values=np.asarray(values)
        if len(set(ids.tolist()))!=len(ids) or (ids<0).any() or (ids>=n_cases).any() or (assignment[ids]>=0).any():
            raise ValueError('duplicate or invalid OOF cases')
        if values.shape!=(len(ids),*shape):raise ValueError('OOF prediction shape mismatch')
        prediction[ids]=values;assignment[ids]=fold
    if (assignment<0).any():raise ValueError('missing OOF predictions')
    return prediction,assignment


def train_fold_isolated(config_path):
    """Give each fold a fresh CUDA process; preserve the exact training config."""
    from . import blade_pipeline as bp
    subprocess.run([sys.executable,'-u',str(bp.ROOT/'run_single_blade.py'),'stage3',
                    '--config',str(Path(config_path).resolve())],cwd=bp.ROOT,check=True,
                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))


def run_cross_validation(cfg):
    import torch
    from . import blade_pipeline as bp
    work=bp.ROOT/cfg['work'];options=cfg['cross_validation'];out=work/options['output']
    original=work/'stage2/paired_data.npz';lf=work/'stage1/best.pt'
    selected=bp.read_json(bp.ROOT/'configs/selected_model.json')
    selected_path=bp.ROOT/selected['checkpoint'];selected_hash=bp.digest(selected_path)
    identity=dict(paired_data_sha256=bp.digest(original),lf_sha256=bp.digest(lf),config=cfg)
    if out.exists():
        if bp.read_json(out/'experiment.json')!=identity:raise ValueError('CV config or frozen inputs changed')
    else:
        out.mkdir();bp.save_json(out/'experiment.json',identity)
    with np.load(original) as f:arrays={k:f[k] for k in f.files}
    if len(arrays['case_ids'])!=cfg['expected_hf_cases']:raise ValueError('unexpected HF case count')
    folds=grouped_cv_splits(arrays['geometry_ids'],arrays['conditions'][:,1],k=options['folds'],seed=cfg['seed'])
    bp.save_json(out/'folds.json',folds)
    if selected_hash!=selected['checkpoint_sha256']:raise ValueError('selected checkpoint mismatch')
    fold_results=[];predictions=[]
    for i,split in enumerate(folds,1):
        asset_dir=out/f'fold_{i:02d}'/'stage2';asset_dir.parent.mkdir(exist_ok=True)
        fold_cfg={k:v for k,v in cfg.items() if k!='cross_validation'}
        fold_cfg.update(flow_assets=asset_dir.relative_to(work).as_posix(),flow_output=f"{options['run_prefix']}_fold{i:02d}")
        run=work/fold_cfg['flow_output']
        if not asset_dir.exists():
            asset_dir.mkdir()
            try:stats,source=fit_fold_source(arrays,split,cfg)
            except Exception as exc:
                bp.save_json(asset_dir/'status.json',dict(status='source_fit_failed',error=str(exc)));raise
            bp.validate_flow_assets(arrays,split,stats,source,split_method='geometry')
            os.link(original,asset_dir/'paired_data.npz')
            for name,value in [('splits',split),('normalization',stats),('source',source)]:bp.save_json(asset_dir/(name+'.json'),value)
            paths=[original,lf,asset_dir/'splits.json',asset_dir/'normalization.json',asset_dir/'source.json']
            bp.save_json(asset_dir/'provenance.json',{str(p.resolve()):bp.digest(p) for p in paths})
            bp.save_json(asset_dir/'status.json',dict(status='complete',fold=i,fit_indices=split['train']))
        if bp.read_json(asset_dir/'splits.json')!=split:raise ValueError('existing fold split mismatch')
        bp.save_json(asset_dir.parent/'config.json',fold_cfg)
        print(json.dumps(dict(event='cv_fold_start',fold=i,total_folds=len(folds),
            counts={k:len(v) for k,v in split.items()},tsr_le_2_counts={k:int(np.sum(arrays['conditions'][v,1]<=2)) for k,v in split.items()},
            flow_output=fold_cfg['flow_output'])),flush=True)
        if (run/'metrics.json').exists():
            ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            if ck['config']!=fold_cfg or ck['splits']!=split:raise ValueError('completed fold config mismatch')
            provenance=bp.read_json(asset_dir/'provenance.json')
            if not all(bp.digest(Path(p))==h for p,h in provenance.items()):raise ValueError('completed fold input identity mismatch')
            result=bp.read_json(run/'metrics.json')
        else:
            if run.exists():raise FileExistsError(f'incomplete fold output retained: {run}')
            train_fold_isolated(asset_dir.parent/'config.json')
            result=bp.read_json(run/'metrics.json')
        ids=np.asarray(split['test'],dtype=int)
        with np.load(run/'test_predictions.npz') as f:
            np.testing.assert_array_equal(f['case_ids'],arrays['case_ids'][ids]);np.testing.assert_array_equal(f['target'],arrays['target'][ids])
            np.testing.assert_array_equal(f['lf'],arrays['lf_prediction'][ids]);np.testing.assert_array_equal(f['theta'],arrays['theta'])
            prediction=f['mean'].copy()
        predictions.append((i,ids,prediction))
        fold_results.append(dict(fold=i,counts={k:len(v) for k,v in split.items()},best_epoch=result['best_epoch'],
            epochs_run=result['epochs_run'],test=error_summary(prediction,arrays['target'][ids])))
        bp.save_json(out/'fold_metrics.json',fold_results)
        print(json.dumps(dict(event='cv_fold_complete',**fold_results[-1])),flush=True)
        if torch.cuda.is_available():torch.cuda.empty_cache()
    mean,fold_id=assemble_oof(len(arrays['case_ids']),predictions,arrays['target'].shape[1:])
    fold_rmse=np.asarray([r['test']['pooled_rmse'] for r in fold_results])
    summary=dict(evaluation='grouped 5-fold outer OOF, inner grouped validation for early stopping',targets=bp.TARGETS,
        scope='HF labels held out by geometry; frozen LF may have seen these geometries; each case tested exactly once',
        fold_rmse_mean=fold_rmse.mean(0).tolist(),fold_rmse_std=fold_rmse.std(0,ddof=1).tolist(),
        all=error_summary(mean,arrays['target']),folds=fold_results,
        robust_metric='per-case RMSE median and IQR; outliers are retained',
        comparison_caveat='OOF100 is a different evaluation population from the previous fixed15 test set')
    for name,mask in [('low_tsr',arrays['conditions'][:,1]<=2),('high_tsr',arrays['conditions'][:,1]>2)]:
        if mask.any():summary[name]=error_summary(mean[mask],arrays['target'][mask])
    np.savez_compressed(out/'oof_predictions.npz',mean=mean,target=arrays['target'],lf=arrays['lf_prediction'],
        theta=arrays['theta'],case_ids=arrays['case_ids'],geometry_ids=arrays['geometry_ids'],conditions=arrays['conditions'],fold=fold_id)
    errors=np.sqrt(np.mean((mean-arrays['target'])**2,axis=1))
    with (out/'per_case_errors.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['case_id','geometry_id','fold','wind_speed','TSR',*bp.TARGETS])
        for i,case in enumerate(arrays['case_ids']):writer.writerow([str(case),str(arrays['geometry_ids'][i]),int(fold_id[i]),*arrays['conditions'][i],*errors[i]])
    if bp.digest(selected_path)!=selected_hash:raise ValueError('selected model changed during CV')
    bp.save_json(out/'summary.json',summary);print(json.dumps(dict(event='cv_complete',summary=summary)),flush=True)
    return summary
