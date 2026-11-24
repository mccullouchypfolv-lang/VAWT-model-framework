"""Combine native single-blade HF cases while preserving frozen LF assets."""
from pathlib import Path
import json
import numpy as np
from .data import read_json,save_json,digest


def collect_hf_folders(cfg,root):
    roots=cfg.get('hf_roots')
    if roots is None:roots=[cfg['hf_root']]
    if not isinstance(roots,list) or not roots or any(not isinstance(p,str) or not p for p in roots):
        raise ValueError('hf_roots must be a nonempty list of directories')
    found={}
    for value in roots:
        folder=(Path(root)/value).resolve()
        if not folder.is_dir():raise FileNotFoundError(folder)
        cases=sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith('S'))
        if not cases:raise ValueError('no HF case directories: '+str(folder))
        for case in cases:
            if case.name in found:raise ValueError('duplicate HF case identity: '+case.name)
            found[case.name]=case
    expected=cfg.get('expected_hf_cases',49)
    if isinstance(expected,bool) or not isinstance(expected,int) or expected<1 or len(found)!=expected:
        raise ValueError(f'expected {expected} unique HF cases, found {len(found)}')
    return [found[name] for name in sorted(found)]


def build_hf_arrays(folders,cache,manifest):
    from .blade_data import read_hf_case,match_cases
    cache=Path(cache);lookup={r['case_id']:r for r in manifest['records']}
    if len({p.name for p in folders})!=len(folders):raise ValueError('duplicate HF case identity')
    missing=sorted({p.name for p in folders}-lookup.keys())
    if missing:raise ValueError('HF cases missing completed LF inputs: '+str(missing))
    samples=[];provenance={}
    for folder in folders:
        record=lookup[folder.name];path=cache/record['file']
        if digest(path)!=record['cache_hash']:raise ValueError('LF cache hash mismatch: '+folder.name)
        item=read_hf_case(folder,manifest['config'])
        with np.load(path,allow_pickle=False) as data:
            lf=dict(case_id=folder.name,conditions=data['a'],theta=data['theta'],reference_conditions=data['reference_conditions'])
        match_cases(lf,item)
        if item['y'].shape!=(160,3) or not np.isfinite(item['y']).all():raise ValueError('invalid HF target: '+folder.name)
        if samples and not np.allclose(item['theta'],samples[0]['theta'],atol=1e-6,rtol=0):raise ValueError('HF grids differ')
        samples.append(item)
        provenance[folder.name]=dict(item['metadata'],source_directory=str(folder.resolve()))
    if not samples:raise ValueError('empty HF dataset')
    arrays=dict(case_ids=np.array([p.name for p in folders]),
        geometry_ids=np.array([lookup[p.name]['geometry_id'] for p in folders]),
        conditions=np.stack([s['conditions'] for s in samples]),theta=samples[0]['theta'],
        target=np.stack([s['y'] for s in samples]))
    return arrays,provenance


def prepare_hf(cfg):
    """Rebuild only HF targets using an existing verified LF cache/checkpoint."""
    import torch
    from .blade_pipeline import ROOT,TARGETS,validate_parameter_contract
    from .blade_data import INPUT_CONTRACT
    work=ROOT/cfg['work'];cache=work/'lf_cache'
    outputs=[work/n for n in ('hf_data.npz','hf_provenance.json','preparation.json')]
    for path in outputs:
        if path.exists():raise FileExistsError(path)
    manifest=read_json(cache/'manifest.json')
    checkpoint=torch.load(work/'stage1/best.pt',map_location='cpu',weights_only=False)
    validate_parameter_contract(checkpoint,manifest)
    if digest(cache/'manifest.json')!=checkpoint['manifest_sha256']:raise ValueError('LF manifest differs from frozen checkpoint')
    folders=collect_hf_folders(cfg,ROOT)
    arrays,provenance=build_hf_arrays(folders,cache,manifest)
    work.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(outputs[0],**arrays);save_json(outputs[1],provenance)
    sources={str(p.parent):sum(q.parent==p.parent for q in folders) for p in folders}
    result=dict(status='prepared',lf_cases=len(manifest['records']),hf_cases=len(folders),targets=TARGETS,
        input_contract=INPUT_CONTRACT,hf_sources=sources,frozen_lf_reused=True,
        lf_checkpoint_sha256=digest(work/'stage1/best.pt'),manifest_sha256=digest(cache/'manifest.json'))
    save_json(outputs[2],result);print(json.dumps(result),flush=True)
    return result
