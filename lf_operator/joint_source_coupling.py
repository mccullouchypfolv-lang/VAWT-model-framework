"""v2 empirical source/coupling design; v1 NAGO and kFFM code stays unchanged.

Source search is finite. Coupling search is restricted to the convex segment
between entropic kernel OT and the independent product plan. The condition
penalty is a finite-source-atom independence diagnostic, not an infinite-
dimensional conditional measure guarantee.
"""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from .kernel_ot import kernel_cost,sinkhorn_log,draw_plan_pairs,SinkhornConvergenceError
from .spectral_source import sample_source


def _weight(value,name,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0 or (positive and value==0):
        raise ValueError(f'{name} must be finite and '+('positive' if positive else 'nonnegative'))
    return float(value)


def conditional_chi2(plan,condition_ids):
    """Normalized chi-square divergence from empirical source/condition independence."""
    p=np.asarray(plan,dtype=float);labels=np.asarray(condition_ids)
    if p.ndim!=2 or min(p.shape)<1 or labels.shape!=(p.shape[1],) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1.,atol=1e-6):
        raise ValueError('a probability plan and one condition ID per target are required')
    _,inverse,counts=np.unique(labels,return_inverse=True,return_counts=True)
    groups=len(counts)
    if groups==1:return 0.
    joint=np.stack([p[:,inverse==h].sum(1) for h in range(groups)],axis=1)
    independent=(np.ones(p.shape[0])/p.shape[0])[:,None]*(counts/p.shape[1])[None,:]
    return float(np.sum((joint-independent)**2/independent)/(groups-1))


def regularize_coupling(cost,ot_plan,condition_ids,*,epsilon=.02,transport_weight=1.,condition_weight=1.):
    """Minimize the stated objective over pi=(1-alpha)*pi_OT+alpha*(a x b)."""
    epsilon=_weight(epsilon,'epsilon',positive=True)
    tw=_weight(transport_weight,'transport_weight',positive=True)
    cw=_weight(condition_weight,'condition_weight')
    cost=np.asarray(cost,dtype=float);p=np.asarray(ot_plan,dtype=float)
    if cost.shape!=p.shape or not np.isfinite(cost).all() or (cost<0).any():raise ValueError('invalid kernel cost')
    before=conditional_chi2(p,condition_ids)
    product=np.full_like(p,1/p.size)
    def evaluate(alpha):
        mix=(1-alpha)*p+alpha*product
        positive=mix>0
        kl=float(np.sum(mix[positive]*np.log(mix[positive]/product[positive])))
        expected=float(np.sum(mix*cost));condition=(1-alpha)**2*before
        transport=expected+epsilon*kl
        return tw*transport+cw*condition,expected,kl,condition,transport
    candidates=[0.,1.] if cw else [0.]
    if cw and before>1e-15:
        result=minimize_scalar(lambda x:evaluate(x)[0],bounds=(0.,1.),method='bounded',options=dict(xatol=1e-6,maxiter=100))
        if not result.success:raise RuntimeError('conditional coupling mixture optimization failed')
        candidates.append(float(result.x))
    alpha=min(candidates,key=lambda x:evaluate(x)[0])
    objective,expected,kl,after,transport=evaluate(alpha)
    plan=(1-alpha)*p+alpha*product
    error=max(np.max(abs(plan.sum(1)-1/len(plan))),np.max(abs(plan.sum(0)-1/plan.shape[1])))
    return plan,dict(independent_fraction=alpha,condition_chi2_before=before,condition_chi2_after=after,
        kernel_transport_cost=expected,transport_kl=kl,regularized_transport=transport,
        joint_objective=objective,ot_endpoint_objective=evaluate(0.)[0],independent_endpoint_objective=evaluate(1.)[0],
        marginal_error=float(error),coupling_family='convex OT/product mixture',conditional_source_independence_guaranteed=False)


def couple_joint_endpoints(source,target,condition_ids,*,transport_weight=1.,condition_weight=1.,
                          kernel='rbf',epsilon=.05,max_iter=2000,tol=1e-8,count=None,seed=None,rng=None,fixed_metric=None,**kernel_options):
    if fixed_metric is not None:
        if set(fixed_metric)!={'bandwidth','median_normalize'} or fixed_metric['median_normalize'] is not False:
            raise ValueError('joint metric requires fixed bandwidth and unnormalized bounded kernel cost')
        kernel_options.update(fixed_metric)
    cost,metadata=kernel_cost(source,target,kernel=kernel,**kernel_options)
    solved=sinkhorn_log(cost,epsilon=epsilon,max_iter=max_iter,tol=tol)
    plan,regularization=regularize_coupling(cost,solved['plan'],condition_ids,epsilon=epsilon,
        transport_weight=transport_weight,condition_weight=condition_weight)
    if regularization['marginal_error']>tol+1e-12:raise RuntimeError('mixture marginal tolerance violated')
    i,j=draw_plan_pairs(plan,len(source) if count is None else count,seed=seed,rng=rng)
    diagnostics=dict(metadata,**{k:v for k,v in solved.items() if k!='plan'})
    diagnostics.update(regularization)
    return dict(source_indices=i,target_indices=j,plan=plan,cost=cost,diagnostics=diagnostics)


def _multiplicity(n):
    w=np.full(n//2+1,2.);w[0]=1.
    if n%2==0:w[-1]=1.
    return w


def spectral_mismatch(source):
    losses=[]
    for c in source['channels']:
        target=np.asarray(c['target_psd'],dtype=float);psd=np.asarray(c['psd'],dtype=float)
        n=int(c['resolved_modes']);floor=max(float(target.max())*1e-12,np.finfo(float).tiny)
        ratio=np.maximum(target[1:n+1],floor)/np.maximum(psd[1:n+1],floor)
        losses.append(float(np.mean(np.log(ratio)**2)))
    return float(np.mean(losses))


def source_candidates(source,*,nu_factors=(.5,1.),kappa_factors=(1.,2.,4.)):
    for values,name in [(nu_factors,'nu_factors'),(kappa_factors,'kappa_factors')]:
        if not values:raise ValueError(f'{name} must not be empty')
        for value in values:_weight(value,name,positive=True)
    candidates=[dict(name='v1_source',source=copy.deepcopy(source))]
    seen={tuple((round(c['nu'],12),round(c['kappa'],12)) for c in source['channels'])}
    for nf in nu_factors:
        for kf in kappa_factors:
            candidate=copy.deepcopy(source)
            for original,c in zip(source['channels'],candidate['channels']):
                nu=min(original['nu']*nf,original['nu_upper']*(1-1e-6))
                kappa=float(np.clip(original['kappa']*kf,.3,30.))
                if nu<=0:raise ValueError('candidate is outside admissible nu range')
                k=np.arange(c['n_theta']//2+1,dtype=float);w=_multiplicity(c['n_theta'])
                shape=(kappa*kappa+k*k)**(-nu-.5)
                previous=np.asarray(original['psd'],dtype=float)
                positive_variance=float(w[1:]@previous[1:])
                amplitude=positive_variance/float(w[1:]@shape[1:])
                psd=amplitude*shape;psd[0]=previous[0]
                c.update(nu=float(nu),kappa=kappa,amplitude=float(amplitude),psd=psd.tolist())
                diagnostic=c.setdefault('diagnostics',{})
                target=np.asarray(c['target_psd']);floor=max(target.max()*1e-12,np.finfo(float).tiny)
                mu=np.maximum(target,floor)/np.maximum(psd,floor)
                diagnostic.update(source_total_variance=float(w@psd),mu=mu.tolist(),
                    linear_bridge_L=(.5*(np.sqrt(mu)+1/np.sqrt(mu))).tolist(),
                    log_spectral_mse=float(np.mean(np.log(mu[1:c['resolved_modes']+1])**2)),
                    total_variance_difference=float(w@(psd-target)),joint_candidate=True)
            key=tuple((round(c['nu'],12),round(c['kappa'],12)) for c in candidate['channels'])
            if key not in seen:
                seen.add(key);candidates.append(dict(name=f'nu_x{nf:g}_kappa_x{kf:g}',source=candidate))
    return candidates


def select_joint_source(arrays,splits,stats,source,cfg,pairing,*,seed=42,progress=None):
    """Offline finite eta search with common random numbers; train labels only."""
    from .blade_pipeline import encode_residual,coupling_method
    coupling_method(pairing,joint_enabled=True)
    pairing={k:v for k,v in pairing.items() if k!='method'}
    batches=cfg.get('selection_batches',8);batch_size=cfg.get('batch_size',256)
    for value in [batches,batch_size]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('positive integer selection budget required')
    sw=_weight(cfg.get('spectral_weight',1.),'spectral_weight')
    tw=_weight(cfg.get('transport_weight',1.),'transport_weight',positive=True)
    cw=_weight(cfg.get('condition_weight',1.),'condition_weight')
    rng=np.random.default_rng(seed+70000)
    batch_ids=[rng.choice(splits['train'],batch_size,replace=True) for _ in range(batches)]
    source_seeds=rng.integers(0,2**32,size=batches).tolist()
    # One metric for all eta, all selection batches and all online batches.
    # Baseline-source calibration uses the same first training batch/seed.
    random=np.random.default_rng(source_seeds[0])
    calibration=np.concatenate([sample_source(c,batch_size,rng=random) for c in source['channels']],axis=-1).astype('float32')
    cost_options={k:v for k,v in pairing.items() if k not in ('epsilon','max_iter','tol')}
    cost_options['median_normalize']=False
    _,metric=kernel_cost(calibration,encode_residual(arrays['residual'][batch_ids[0]],stats),**cost_options)
    fixed_metric=dict(bandwidth=metric['bandwidth'],median_normalize=False)
    candidates=source_candidates(source,nu_factors=cfg.get('nu_factors',[.5,1.]),kappa_factors=cfg.get('kappa_factors',[1.,2.,4.]))
    records=[];best=None
    for i,item in enumerate(candidates):
        candidate=item['source'];terms=[]
        for b,ids in enumerate(batch_ids):
            random=np.random.default_rng(source_seeds[b])
            g0=np.concatenate([sample_source(c,batch_size,rng=random) for c in candidate['channels']],axis=-1).astype('float32')
            target=encode_residual(arrays['residual'][ids],stats)
            try:
                result=couple_joint_endpoints(g0,target,arrays['coupling_condition_ids'][ids],transport_weight=tw,
                    condition_weight=cw,count=0,seed=seed+b,fixed_metric=fixed_metric,**pairing)
            except SinkhornConvergenceError as exc:
                terms=[];failure=str(exc);break
            terms.append(result['diagnostics'])
        entry=dict(index=i,name=item['name'],parameters=[{k:c[k] for k in ['nu','kappa','amplitude']} for c in candidate['channels']])
        if not terms:
            entry.update(status='sinkhorn_failed',reason=failure);records.append(entry)
            if progress:progress(entry)
            continue
        spec=spectral_mismatch(candidate)
        transport=float(np.mean([t['regularized_transport'] for t in terms]))
        cond=float(np.mean([t['condition_chi2_after'] for t in terms]))
        score=sw*spec+tw*transport+cw*cond
        entry.update(status='complete',spectral_loss=spec,regularized_transport=transport,condition_chi2=cond,
            condition_chi2_before=float(np.mean([t['condition_chi2_before'] for t in terms])),
            mean_independent_fraction=float(np.mean([t['independent_fraction'] for t in terms])),objective=score)
        records.append(entry)
        if best is None or score<best[0]:best=(score,i,candidate)
        if progress:progress(entry)
    if best is None:raise RuntimeError('No joint source candidate passed Sinkhorn convergence checks')
    _,index,selected=best
    report=dict(version='v2',selected_index=index,selected_name=candidates[index]['name'],fit_indices=list(splits['train']),
        selection_batches=batches,batch_size=batch_size,seed=seed,weights=dict(spectral=sw,transport=tw,condition=cw),
        fixed_metric=fixed_metric,metric_policy='calibrate once on v1 source and first train-only batch; no cost median rescaling',
        candidates=records,variance_policy='preserve each v1 source channel total variance and DC variance',
        scope='finite eta candidates and convex OT/product coupling family; empirical conditional chi-square proxy')
    selected=copy.deepcopy(selected)
    selected['joint_design']=dict(version='v2',selected_name=report['selected_name'],weights=report['weights'])
    return selected,report


def condition_identities(arrays):
    """Group by declared physical inputs, never by responses or case IDs."""
    conditions=np.asarray(arrays['conditions']);geometry=np.asarray(arrays['geometry'])
    if conditions.ndim!=2 or conditions.shape[1]!=2 or geometry.shape!=(len(conditions),20,2):
        raise ValueError('expected U/TSR and 20-layer chord/thickness inputs')
    inputs=np.concatenate([conditions,geometry.reshape(len(conditions),-1)],axis=1)
    if not np.isfinite(inputs).all():raise ValueError('nonfinite physical condition inputs')
    return np.unique(inputs,axis=0,return_inverse=True)[1]


def design_settings(cfg):
    joint=cfg.get('joint_source',{})
    if joint.get('enabled') is not True:raise ValueError('joint_source.enabled must be true for v2')
    return dict(joint_source=joint,pairing=cfg['pairing'],batch_size=cfg['batch_size'],seed=cfg['seed'])


def joint_directory(cfg):
    from .blade_pipeline import ROOT
    name=cfg.get('joint_source',{}).get('output','stage2_joint_v2')
    if not isinstance(name,str) or not name or name in ('.','..','stage1','stage2','lf_cache',cfg.get('flow_output','stage3')) or any(c in name for c in '/\\:'):
        raise ValueError('joint source output must be a distinct directory name')
    return ROOT/cfg['work']/name


def load_physical_arrays(arrays,cfg):
    from .blade_pipeline import ROOT
    from .conditioning import geometry_from_cache
    from .data import read_json
    work=ROOT/cfg['work']
    # Verify geometry against the frozen LF cache even when paired assets have it.
    result=geometry_from_cache(arrays,work/'lf_cache',read_json(work/'lf_cache/manifest.json'))
    result['coupling_condition_ids']=condition_identities(result)
    return result


def prepare_joint_source(cfg):
    from .blade_pipeline import ROOT,validate_flow_assets
    from .data import read_json,save_json,digest
    settings=design_settings(cfg);work=ROOT/cfg['work'];baseline=work/'stage2';out=joint_directory(cfg)
    if out.exists():raise FileExistsError(out)
    if read_json(baseline/'status.json')['status']!='complete':raise ValueError('baseline stage2 incomplete')
    provenance=read_json(baseline/'provenance.json')
    if not all(digest(Path(p))==h for p,h in provenance.items()):raise ValueError('frozen stage-two inputs changed')
    with np.load(baseline/'paired_data.npz') as data:arrays={k:data[k] for k in data.files}
    splits=read_json(baseline/'splits.json');stats=read_json(baseline/'normalization.json');source=read_json(baseline/'source.json')
    validate_flow_assets(arrays,splits,stats,source,split_method=cfg.get('hf_split',{}).get('method','geometry'))
    arrays=load_physical_arrays(arrays,cfg)
    selection_cfg=dict(settings['joint_source'],batch_size=cfg['batch_size'])
    selected,report=select_joint_source(arrays,splits,stats,source,selection_cfg,cfg['pairing'],seed=cfg['seed'],
        progress=lambda row:print(json.dumps(dict(stage='stage2-v2',candidate=row)),flush=True))
    out.mkdir()
    save_json(out/'source.json',selected);save_json(out/'selection.json',report);save_json(out/'settings.json',settings)
    inputs=[baseline/'provenance.json',baseline/'source.json',baseline/'paired_data.npz',baseline/'splits.json',baseline/'normalization.json',
            work/'stage1/best.pt',work/'lf_cache/manifest.json']+[out/n for n in ('source.json','selection.json','settings.json')]
    save_json(out/'provenance.json',{str(p.resolve()):digest(p) for p in inputs})
    status=dict(status='complete',version='v2',fit_indices=splits['train'],selected_name=report['selected_name'],
        condition_definition='U, TSR, full 20-layer chord and thickness profiles',conditional_source_independence_guaranteed=False)
    save_json(out/'status.json',status);print(json.dumps(status),flush=True)
    return status


def load_joint_source(cfg,arrays):
    from .data import read_json,digest
    out=joint_directory(cfg)
    if read_json(out/'status.json')['status']!='complete':raise ValueError('joint source preparation incomplete')
    if read_json(out/'settings.json')!=design_settings(cfg):raise ValueError('joint source settings changed; prepare a separate v2 source directory')
    provenance=read_json(out/'provenance.json')
    if not all(digest(Path(p))==h for p,h in provenance.items()):raise ValueError('joint source inputs changed')
    arrays=load_physical_arrays(arrays,cfg)
    options={k:cfg['joint_source'].get(k,1.) for k in ('transport_weight','condition_weight')}
    options['fixed_metric']=read_json(out/'selection.json')['fixed_metric']
    return read_json(out/'source.json'),arrays,options
