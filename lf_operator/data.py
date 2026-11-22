"""Shared file utilities and geometry-group dataset splitting."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_split(records, seed=1492, val_fraction=.15, test_fraction=.15):
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction+test_fraction >= 1:
        raise ValueError('invalid split fractions')
    groups = np.array(sorted({r['geometry_id'] for r in records}))
    if len(groups) < 3:
        raise ValueError('need at least three distinct geometry groups')
    np.random.default_rng(seed).shuffle(groups)
    nv, nt = max(1,round(len(groups)*val_fraction)), max(1,round(len(groups)*test_fraction))
    if nv+nt >= len(groups):
        raise ValueError('split leaves no training geometries')
    assignment = {g: s for s, gs in [('val',groups[:nv]),('test',groups[nv:nv+nt]),('train',groups[nv+nt:])] for g in gs}
    return {s:[i for i,r in enumerate(records) if assignment[r['geometry_id']] == s] for s in ('train','val','test')}
