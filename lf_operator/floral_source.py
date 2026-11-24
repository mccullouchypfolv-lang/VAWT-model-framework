"""Fixed zero-mean FLORAL GP source, without residual-spectrum fitting.

Upstream9d1c1c: gp/gp.py nu=.5; flow/flow.py residual covariance factor2,
confidence=0; examples/oneDCorr/config.yml lengthscale=.001/outputscale1.
Use normalized angular coordinate x=theta/(2*pi) with the ordinary Euclidean
Matérn kernel. This intentionally does not periodicize the upstream prior.
"""
import math
import numbers
import numpy as np

KIND='floral_fixed_matern_gp'
SCOPE='fixed FLORAL Matern GP on normalized angular coordinates; Euclidean kernel; evaluated at query grid'
COMMIT='9d1c1c0c25a7b367ad24441358660e98b274aa2c'


def floral_matern_source(n_theta,*,lengthscale=.001,outputscale=1.,residual_covariance_factor=2.):
    if isinstance(n_theta,(bool,np.bool_)) or not isinstance(n_theta,(int,np.integer)) or n_theta<4:
        raise ValueError('n_theta must be integer >=4')
    for name,value in [('lengthscale',lengthscale),('outputscale',outputscale),('residual_covariance_factor',residual_covariance_factor)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not math.isfinite(value) or value<=0:
            raise ValueError(name+' must be finite positive')
    return dict(kind=KIND,n_theta=int(n_theta),n_channels=1,nu=.5,lengthscale=float(lengthscale),
        outputscale=float(outputscale),residual_covariance_factor=float(residual_covariance_factor),confidence=0.,
        mean=0.,domain=[0.,2*math.pi],coordinate='theta/(2*pi)',periodic_kernel=False,
        marginal_variance=float(outputscale*residual_covariance_factor),parameter_fit='fixed; no HF spectrum fitted',
        upstream_commit=COMMIT,upstream_gp=f'https://github.com/sahilbhola14/floral/blob/{COMMIT}/src/floral/gp/gp.py',
        upstream_flow=f'https://github.com/sahilbhola14/floral/blob/{COMMIT}/src/floral/flow/flow.py')


def _parameters(spec,theta):
    if spec.get('kind')!=KIND or spec.get('nu')!=.5 or spec.get('confidence')!=0. or spec.get('coordinate')!='theta/(2*pi)':
        raise ValueError('invalid fixed FLORAL GP specification')
    validated=floral_matern_source(spec['n_theta'],lengthscale=spec['lengthscale'],outputscale=spec['outputscale'],
                                  residual_covariance_factor=spec['residual_covariance_factor'])
    if theta is None:
        from .resolution import midpoint_theta
        theta=midpoint_theta(spec['n_theta'])
    theta=np.asarray(theta,dtype=float)
    if theta.ndim!=1 or len(theta)<4 or not np.isfinite(theta).all() or np.any(theta<0) or np.any(theta>=2*math.pi) or np.any(np.diff(theta)<=0):
        raise ValueError('ordered distinct angular nodes in [0,2*pi) required')
    return theta/(2*math.pi),validated['lengthscale'],validated['marginal_variance']


def floral_covariance(spec,theta=None):
    x,lengthscale,variance=_parameters(spec,theta)
    return variance*np.exp(-np.abs(x[:,None]-x[None,:])/lengthscale)


def sample_floral_source(spec,count,*,seed=None,rng=None,theta=None):
    """Exact exponential-kernel GP using a stationary Gaussian Markov chain.

    Channels are sampled independently by the existing multichannel dispatcher.
    Query-grid sampling preserves the kernel, not the seeded finite realization.
    """
    if isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or count<1:
        raise ValueError('count must be positive integer')
    if seed is not None and rng is not None:raise ValueError('pass seed or rng, not both')
    if rng is None:rng=np.random.default_rng(seed)
    if not isinstance(rng,np.random.Generator):raise ValueError('rng must be numpy Generator')
    x,lengthscale,variance=_parameters(spec,theta)
    draws=rng.normal(size=(count,len(x)))
    decay=np.exp(-np.diff(x)/lengthscale)
    innovation=np.sqrt(variance*(-np.expm1(-2*np.diff(x)/lengthscale)))
    samples=np.empty_like(draws);samples[:,0]=math.sqrt(variance)*draws[:,0]
    for i in range(1,len(x)):samples[:,i]=decay[i-1]*samples[:,i-1]+innovation[i-1]*draws[:,i]
    return samples[...,None]


def prepare_floral_assets(cfg):
    """Keep frozen default targets/split/normalization; replace only the prior."""
    import shutil
    from pathlib import Path
    from . import blade_pipeline as bp
    work=bp.ROOT/cfg['work'];original=work/'stage2'
    out=work/cfg.get('flow_assets','stage2_floral_fixed')
    if out.resolve()==original.resolve() or out.exists():raise FileExistsError('fixed source requires a new asset directory')
    if bp.read_json(original/'status.json')['status']!='complete':raise ValueError('original paired assets are incomplete')
    provenance=bp.read_json(original/'provenance.json')
    if not all(bp.digest(Path(p))==h for p,h in provenance.items()):raise ValueError('frozen original inputs changed')
    with np.load(original/'paired_data.npz',allow_pickle=False) as d:arrays={k:d[k] for k in d.files}
    splits=bp.read_json(original/'splits.json');stats=bp.read_json(original/'normalization.json')
    options=dict(cfg['source_design'])
    if options.pop('method')!='floral_fixed_matern':raise ValueError('unsupported fixed GP source design')
    channels=[floral_matern_source(len(arrays['theta']),**options) for _ in range(3)]
    source=dict(channels=channels,targets=bp.TARGETS,target_contract=bp.TARGET_CONTRACT,input_contract=bp.INPUT_CONTRACT,
        fit_indices=splits['train'],fit_indices_role='data-split binding only; GP parameters fixed without HF fitting',
        split_method='geometry',cross_channel_covariance='independent draws from the same fixed GP',
        source_design='fixed official FLORAL residual prior',nago_spectrum_fit=False)
    if cfg.get('hf_split',{}).get('method','geometry')!='geometry':raise ValueError('this source ablation restores the original geometry split')
    bp.validate_flow_assets(arrays,splits,stats,source,split_method='geometry')
    frozen=[work/'stage1/best.pt',original/'paired_data.npz',original/'normalization.json',original/'splits.json',original/'source.json']
    identity={str(p.resolve()):bp.digest(p) for p in frozen}
    out.mkdir()
    for name in ('paired_data.npz','normalization.json','splits.json','split_design.json'):
        if (original/name).exists():shutil.copy2(original/name,out/name)
    bp.save_json(out/'source.json',source)
    bp.save_json(out/'source_design.json',dict(**cfg['source_design'],upstream_commit=COMMIT,
        coordinate='theta/(2*pi)',periodic_kernel=False,source_scope=SCOPE,
        changes='source only; no bridge/no network/no normalization/no split changes'))
    for name in ('paired_data.npz','normalization.json','splits.json','source.json','source_design.json'):
        p=out/name;identity[str(p.resolve())]=bp.digest(p)
    bp.save_json(out/'provenance.json',identity)
    report=dict(status='complete',source_design='floral_fixed_matern',
        splits={k:len(v) for k,v in splits.items()},marginal_variance=channels[0]['marginal_variance'],
        nago_admissibility='not asserted: deliberately replacing the NAGO design',upstream_commit=COMMIT)
    bp.save_json(out/'status.json',report)
    print(__import__('json').dumps(report),flush=True)
    return report
