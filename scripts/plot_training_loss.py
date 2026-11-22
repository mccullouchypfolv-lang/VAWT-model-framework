"""Plot current parameter-only LF training and validation loss."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/single_blade_parameters_v1/stage1'))
    args=parser.parse_args()
    history=json.loads((args.run/'history.json').read_text(encoding='utf-8'))
    if not history:raise ValueError('no completed epochs')
    best=min(history,key=lambda row:row['val_loss'])
    fig,ax=plt.subplots(figsize=(7,4.2),layout='constrained')
    ax.semilogy([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Training',color='#5B0DAD')
    ax.semilogy([r['epoch'] for r in history],[r['val_loss'] for r in history],label='Validation',color='#3D78C2')
    ax.axvline(best['epoch'],ls='--',color='#555555',lw=1,label=f"Best epoch {best['epoch']}")
    ax.set(xlabel='Epoch',ylabel='Standardized joint loss',title='Parameter-only single-blade LF model')
    ax.legend(frameon=False)
    output=args.run/'figures';output.mkdir(exist_ok=True)
    fig.savefig(output/'training_loss.png',dpi=300);plt.close(fig)


if __name__=='__main__':main()
