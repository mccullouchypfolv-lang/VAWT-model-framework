import unittest
from unittest.mock import patch
import numpy as np
import torch
from lf_operator import blade_pipeline as pipeline


class FlowValidationTests(unittest.TestCase):
    def test_validation_is_fixed_timeweighted_and_excludes_training_targets(self):
        class Zero(torch.nn.Module):
            def forward(self,state,time,condition):return torch.zeros_like(state)
        arrays={'residual':np.array([np.full((8,3),x) for x in (90.,2.,3.,100.)])}
        stats={'mean_profile':np.zeros((8,3)).tolist(),'scale':[1.,1.,1.]}
        condition={'dummy':torch.arange(4)}
        def prepare(model,tensors,ids):return {k:v[ids] for k,v in tensors.items()}
        seed=123;ids=np.tile([1,2],3);rng=np.random.default_rng(seed);expected=[]
        for start in range(0,len(ids),2):
            chosen=ids[start:start+2]
            time=rng.uniform(0,1,(len(chosen),1,1)).astype('float32')
            expected.extend(((1+2*time[:,0,0]**2)*np.mean(arrays['residual'][chosen]**2,axis=(1,2))).tolist())
        args=(Zero(),condition,arrays,[1,2],{},stats)
        options=dict(draws=3,batch_size=2,seed=seed,device='cpu',time_weighting='floral_quadratic')
        with patch.object(pipeline,'sample_multichannel_source',side_effect=lambda s,n,r:np.zeros((n,8,3),dtype='float32')),patch('lf_operator.conditioning.prepare_condition',side_effect=prepare):
            score=pipeline.validation_velocity_mse(*args,**options)
            arrays['residual'][[0,3]]+=10000
            repeated=pipeline.validation_velocity_mse(*args,**options)
        self.assertAlmostEqual(score,float(np.mean(expected)),places=5)
        self.assertEqual(score,repeated)

    def test_scheduler_steps_only_on_tenth_epoch(self):
        p=torch.nn.Parameter(torch.tensor([1.]))
        optimizer,scheduler=pipeline.flow_optimization([p],{'flow_lr':.001,'flow_lr_gamma':.99})
        for epoch in range(1,22):
            optimizer.zero_grad();p.square().sum().backward();optimizer.step()
            pipeline.step_flow_scheduler(scheduler,epoch,10)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],.001*.99**(epoch//10))
        self.assertEqual(scheduler.last_epoch,2)

    def test_scheduler_rejects_invalid_frequency(self):
        for frequency in (0,-1,True,1.5):
            with self.subTest(frequency=frequency),self.assertRaises(ValueError):
                pipeline.step_flow_scheduler(None,1,frequency)


if __name__=='__main__':unittest.main()
