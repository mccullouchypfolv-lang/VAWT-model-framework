"""Inspect conditional residual FM losses and LF/HF/prediction comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_case(axes,data,index,legend=False):
    angle=np.rad2deg(data['theta'])
    for c,ax in enumerate(axes):
        ax.plot(angle,data['lf'][index,:,c],color='#3D78C2',ls='--',lw=1.5,label='LF baseline')
        ax.plot(angle,data['target'][index,:,c],color='#222222',lw=2.,label='HF truth')
        ax.plot(angle,data['mean'][index,:,c],color='#D55E00',ls='-.',lw=1.7,label='FM prediction (mean)')
        ax.set(xlabel='Azimuth (degrees)',ylabel=['Cl','Cd','CM'][c],xlim=(0,360))
        ax.set_xticks([0,90,180,270,360])
        if legend and c==0:ax.legend(frameon=False,fontsize=8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/single_blade_parameters_v1/stage3'))
    args=parser.parse_args();out=args.run/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.linewidth':1.,'xtick.direction':'out','ytick.direction':'out'})
    history=json.loads((args.run/'history.json').read_text(encoding='utf-8'))
    metrics=json.loads((args.run/'metrics.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    axes[0].plot([r['epoch'] for r in history],[r['velocity_mse'] for r in history],color='#5B0DAD')
    axes[0].set(xlabel='Epoch',ylabel='Normalized velocity MSE',title='Training objective')
    valid=[r for r in history if 'val_standardized_curve_mse' in r]
    axes[1].plot([r['epoch'] for r in valid],[r['val_standardized_curve_mse'] for r in valid],color='#3D78C2')
    axes[1].axvline(metrics['best_epoch'],ls='--',color='#555555',label=f"Best epoch {metrics['best_epoch']}")
    axes[1].set(xlabel='Epoch',ylabel='Normalized curve MSE',title='Validation: fixed source draws')
    axes[1].legend(frameon=False)
    fig.savefig(out/'flow_training_loss.png',dpi=300);plt.close(fig)
    with np.load(args.run/'test_predictions.npz') as data:
        fig,axes=plt.subplots(1,3,figsize=(13.5,4),layout='constrained')
        draw_case(axes,data,0,True)
        fig.suptitle(f"First held-out HF case: {str(data['case_ids'][0])} | best epoch {metrics['best_epoch']}",fontsize=11)
        fig.savefig(out/'first_test_LF_HF_prediction.png',dpi=300);plt.close(fig)
        n=len(data['case_ids'])
        fig,axes=plt.subplots(n,3,figsize=(13.5,2.5*n),layout='constrained',squeeze=False)
        diagnostics=[]
        for i in range(n):
            draw_case(axes[i],data,i,i==0)
            axes[i,0].set_title(str(data['case_ids'][i]),loc='left',fontsize=10)
            error=np.sqrt(np.mean((data['mean'][i]-data['target'][i])**2,axis=0))
            baseline=np.sqrt(np.mean((data['lf'][i]-data['target'][i])**2,axis=0))
            diagnostics.append(dict(case_id=str(data['case_ids'][i]),corrected_rmse=error.tolist(),lf_rmse=baseline.tolist(),
                                    pointwise_90_coverage=float(np.mean((data['target'][i]>=data['q05'][i])&(data['target'][i]<=data['q95'][i])))))
        fig.savefig(out/'all_test_LF_HF_prediction.png',dpi=180);plt.close(fig)
        result=dict(best_epoch=metrics['best_epoch'],epochs_run=metrics['epochs_run'],cases=diagnostics,
                    generated_interval_calibrated=False,targets=[str(s) for s in data['targets']])
        (out/'diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))


if __name__=='__main__':main()
