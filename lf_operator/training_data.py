"""Load parameter-only training inputs, keeping labels outside the forward API."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


class LFData(Dataset):
    def __init__(self,cache,manifest,indices,stats,require_targets=True):
        self.cache=Path(cache); self.manifest=manifest; self.indices=indices
        self.stats=stats; self.require_targets=require_targets

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        config=self.manifest['config']; stats=self.stats
        with np.load(self.cache/self.manifest['records'][self.indices[index]]['file'],allow_pickle=False) as saved:
            x=np.asarray(saved['x'],dtype='float32'); a=np.asarray(saved['a'],dtype='float32')
            if x.ndim!=3 or x.shape[:2]!=(20,2) or a.shape!=(2,):raise ValueError('expected two geometry channels and [U_ref,TSR]')
            result=dict(x=(x-np.asarray(stats['x_mean'],dtype='float32')[None,:,None])/np.asarray(stats['x_std'],dtype='float32')[None,:,None],
                        a=(a-np.asarray(stats['a_mean'],dtype='float32'))/np.asarray(stats['a_std'],dtype='float32'),
                        z=(np.asarray(saved['z'],dtype='float32')-config['rotor_bottom_height'])/config['height'],
                        theta=np.asarray(saved['theta'],dtype='float32'),weights=np.asarray(saved['weights'],dtype='float32'))
            if self.require_targets:
                target=np.asarray(saved['y'],dtype='float32')
                result['y']=(target-np.asarray(stats['y_mean'],dtype='float32'))/np.asarray(stats['y_std'],dtype='float32')
        return {k:torch.from_numpy(v) for k,v in result.items()}


def loader(cache,manifest,indices,stats,batch_size,shuffle=False,require_targets=True):
    generator=None if shuffle else torch.Generator().manual_seed(0)
    return DataLoader(LFData(cache,manifest,indices,stats,require_targets),batch_size=batch_size,
                      shuffle=shuffle,num_workers=0,generator=generator)


def to_device(batch,device):return {k:v.to(device) for k,v in batch.items()}


def forward(model,batch,features=False):
    return model(*(batch[k] for k in ('x','a','z','theta','weights')),return_features=features)
