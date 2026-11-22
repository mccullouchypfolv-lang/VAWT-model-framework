"""Re-evaluate a frozen residual model with a selected ODE solver and fixed draws."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from lf_operator.blade_pipeline import (ROOT,setup,read_json,save_json,digest,metrics,
    draw_predictions,flow_output_directory,validate_velocity_checkpoint,validate_flow_assets)
from lf_operator.conditioning import standardize_conditions,tensor_conditions
from lf_operator.residual_flow import ConditionalVelocity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/single_blade.json')
    parser.add_argument('--method',choices=['heun','dopri5'],default='dopri5')
    parser.add_argument('--atol',type=float,default=1e-5)
    parser.add_argument('--rtol',type=float,default=1e-5)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();cfg=read_json(args.config);device=setup(cfg)
    run=flow_output_directory(cfg);assets=ROOT/cfg['work']/'stage2'
    out=args.output or run/f'solver_{args.method}_a{args.atol:g}_r{args.rtol:g}'
    if out.exists():raise FileExistsError(out)
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    validate_velocity_checkpoint(ck)
    if digest(ROOT/cfg['work']/'stage1/best.pt')!=ck['lf_checkpoint_sha256']:
        raise ValueError('LF checkpoint differs from trained flow condition')
    provenance=read_json(assets/'provenance.json')
    if not all(digest(Path(p))==h for p,h in provenance.items()):raise ValueError('frozen stage-two inputs changed')
    with np.load(assets/'paired_data.npz') as data:arrays={k:data[k] for k in data.files}
    splits=read_json(assets/'splits.json')
    stats=ck['residual_stats'];source=ck['source']
    validate_flow_assets(arrays,splits,stats,source)
    if ck['splits']!=splits:raise ValueError('split differs from frozen checkpoint')
    model=ConditionalVelocity(**ck['model_config']).to(device).eval();model.load_state_dict(ck['model'])
    tensors=tensor_conditions(standardize_conditions(arrays,ck['scalers']),arrays,device)
    baseline=read_json(run/'metrics.json')
    if baseline['best_epoch']!=ck['epoch']:raise ValueError('baseline does not describe the selected checkpoint')
    # Reproduce the checkpoint's evaluation budget and source seeds even when
    # the current config was edited for another training experiment.
    draws=ck['config']['evaluation_draws'];seed=ck['config']['seed']
    baseline_method=ck['config'].get('ode_method','heun')
    report={};out.mkdir(parents=True)
    for split in ('val','test'):
        print(json.dumps(dict(status='evaluating',split=split,method=args.method,atol=args.atol,rtol=args.rtol)),flush=True)
        ids=splits[split];solver={};start=time.perf_counter()
        result=draw_predictions(model,tensors,arrays,ids,source,stats,draws,cfg['ode_steps'],
            seed+(20000 if split=='test' else 10000),method=args.method,atol=args.atol,rtol=args.rtol,solver_stats=solver)
        solver['seconds']=time.perf_counter()-start
        report[split]=dict(corrected=metrics(result['mean'],arrays['target'][ids]),solver=solver,
                           baseline_method=baseline_method,baseline_metrics=baseline[split]['corrected'])
        np.savez_compressed(out/f'{split}_predictions.npz',**result,target=arrays['target'][ids],
            lf=arrays['lf_prediction'][ids],theta=arrays['theta'],case_ids=arrays['case_ids'][ids],targets=np.asarray(ck['targets']))
        if split=='test':
            report['per_case']=[dict(case_id=str(arrays['case_ids'][i]),
                rmse=np.sqrt(np.mean((result['mean'][j]-arrays['target'][i])**2,axis=0)).tolist()) for j,i in enumerate(ids)]
        print(json.dumps({split:report[split]}),flush=True)
        save_json(out/'metrics.json',report)
    report.update(checkpoint_sha256=digest(run/'best.pt'),checkpoint_epoch=ck['epoch'],
        draws_per_case=draws,source_seed=seed,atol=args.atol,rtol=args.rtol,
        tolerance_space='normalized residual',weights_updated=False)
    save_json(out/'metrics.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_residual_results import draw_case
    with np.load(out/'test_predictions.npz') as data:
        fig,axes=plt.subplots(2,3,figsize=(13.5,7.3),layout='constrained')
        for row,case in enumerate(['S000006_L050','S000372_L015']):
            i=int(np.flatnonzero(data['case_ids']==case)[0]);draw_case(axes[row],data,i,row==0)
            for channel in range(3):axes[row,channel].set_title(f'{case} | {args.method}',fontsize=10)
        fig.savefig(out/'LF_HF_prediction.png',dpi=220);plt.close(fig)
    print(json.dumps(dict(status='complete',output=str(out.resolve()))),flush=True)


if __name__=='__main__':main()
