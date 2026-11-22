import unittest
from unittest.mock import patch
import numpy as np
from lf_operator.training_data import LFData


class TrainingDataTests(unittest.TestCase):
    def test_inference_reads_no_labels_or_solver_response_fields(self):
        class Saved:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def __getitem__(self,key):
                allowed=dict(x=np.ones((20,2,160),dtype='float32'),a=np.array([15.,1.5],dtype='float32'),
                             z=np.linspace(62.4,168.8,20,dtype='float32'),theta=np.arange(160,dtype='float32'),weights=np.ones(20,dtype='float32'))
                if key not in allowed:raise AssertionError('unexpected input read: '+key)
                return allowed[key]
        manifest={'records':[{'file':'case.npz'}],'config':{'rotor_bottom_height':59.6,'height':112.}}
        stats=dict(x_mean=[0.,0.],x_std=[1.,1.],a_mean=[0.,0.],a_std=[1.,1.])
        with patch('lf_operator.training_data.np.load',return_value=Saved()):
            item=LFData('.',manifest,[0],stats,require_targets=False)[0]
        self.assertEqual(set(item),{'x','a','z','theta','weights'})
        self.assertEqual(tuple(item['x'].shape),(20,2,160))
        self.assertEqual(tuple(item['a'].shape),(2,))

    def test_velocity_derivative_regularizer_matches_target_not_zero(self):
        import torch
        from lf_operator.objectives import velocity_loss
        target=torch.sin(torch.arange(32)*2*torch.pi/32)[None,:,None].expand(1,-1,3)
        self.assertEqual(float(velocity_loss(target,target,.1)),0.)
        self.assertAlmostEqual(float(velocity_loss(target+1,target,.1)),1.,places=5)


if __name__=='__main__':unittest.main()
