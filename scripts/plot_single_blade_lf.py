"""Compare a held-out LF curve with the stage-one model, on native angle bins."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lf_operator.data import read_json
from lf_operator.blade_pipeline import lf_predict,TARGETS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path('artifacts/single_blade_parameters_v1'))
    parser.add_argument('--case-id')
    args=parser.parse_args()
    torch.set_num_threads(4)
    checkpoint=torch.load(args.work/'stage1'/'best.pt',map_location='cpu',weights_only=False)
    cache=args.work/'lf_cache'; manifest=read_json(cache/'manifest.json')
    test=checkpoint['splits']['test']
    if args.case_id:
        matches=[i for i in test if manifest['records'][i]['case_id']==args.case_id]
        if not matches:raise ValueError('requested case is not in the LF test split')
        index=matches[0]
    else:index=test[0]
    case_id=manifest['records'][index]['case_id']
    prediction=lf_predict(checkpoint,cache,manifest,[index],'cpu')['lf_prediction'][0]
    with np.load(cache/manifest['records'][index]['file']) as item:
        target=item['y']; theta=item['theta']; conditions=item['a']; chord=item['x'][:,0,0]
    rmse=np.sqrt(np.mean((prediction-target)**2,axis=0))
    out=args.work/'stage1'/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':12,
                         'axes.linewidth':1.,'xtick.direction':'out','ytick.direction':'out'})
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2),layout='constrained')
    labels=[r'$C_{L,\mathrm{blade}}$',r'$C_{D,\mathrm{blade}}$',r'$C_{M,\mathrm{blade}}$']
    for channel,ax in enumerate(axes):
        ax.plot(np.rad2deg(theta),target[:,channel],color='#222222',lw=2.1,label='LF data')
        ax.plot(np.rad2deg(theta),prediction[:,channel],color='#D55E00',lw=1.8,ls='--',label='Stage 1 prediction')
        ax.set(xlabel='Azimuth (degrees)',ylabel=labels[channel],xlim=(0,360))
        ax.set_xticks([0,90,180,270,360])
        ax.set_title(f'{TARGETS[channel]}   RMSE = {rmse[channel]:.3g}',fontsize=11)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle(f'Test case {case_id} | U={conditions[0]:g} m/s, TSR={conditions[1]:g}, c={chord.min():g}–{chord.max():g} m | best epoch {checkpoint["epoch"]}',fontsize=11)
    filename=out/f'{case_id}_LF_vs_prediction.png'
    fig.savefig(filename,dpi=300); plt.close(fig)
    np.savez_compressed(out/f'{case_id}_LF_vs_prediction.npz',prediction=prediction,target=target,theta=theta,conditions=conditions,targets=np.array(TARGETS))
    result=dict(case_id=case_id,split='test',selection='first test case' if not args.case_id else 'requested test case',
                best_epoch=checkpoint['epoch'],rmse=dict(zip(TARGETS,rmse.tolist())),figure=str(filename.resolve()))
    (out/f'{case_id}_LF_vs_prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
