import unittest

import numpy as np
import torch

from lf_operator.loss_balancing import fit_loss_weights
from lf_operator.objectives import velocity_loss


class LossBalancingTests(unittest.TestCase):
    def test_train_only_weights_balance_groups_and_channels(self):
        residual=np.arange(5*4*3,dtype=float).reshape(5,4,3)/10
        conditions=np.array([[10,1.5],[11,3],[12,4],[13,5],[14,1.5]])
        options=dict(tsr_threshold=2.,variance_floor=.25,min_weight=.25,max_weight=4.)
        stats=dict(scale=[1.,2.,3.])
        report=fit_loss_weights(residual,conditions,[0,1,2,3],stats,options)
        changed=residual.copy();changed[4]=1e9
        other=fit_loss_weights(changed,conditions,[0,1,2,3],stats,options)
        np.testing.assert_array_equal(report['angular_weights'],other['angular_weights'])
        np.testing.assert_allclose(np.mean(report['angular_weights'],axis=0),1.)
        self.assertAlmostEqual(report['group_weights']['low'],2.)
        self.assertAlmostEqual(report['group_weights']['high'],2/3)
        self.assertEqual(report['fit_indices'],[0,1,2,3])

    def test_weighted_loss_matches_manual_value_and_gradient(self):
        prediction=torch.tensor([[[1.,2.],[3.,4.]],[[2.,3.],[4.,5.]]],requires_grad=True)
        target=torch.zeros_like(prediction);time=torch.tensor([0.,1.])
        sample=torch.tensor([2.,.5]);angular=torch.tensor([[.5,1.5],[1.5,.5]])
        loss=velocity_loss(prediction,target,time=time,time_weighting='floral_quadratic',
                           sample_weights=sample,angular_weights=angular)
        expected=((prediction.square()*angular).mean((1,2))*sample*(1+2*time**2)).mean()
        torch.testing.assert_close(loss,expected)
        loss.backward()
        torch.testing.assert_close(prediction.grad,2*prediction.detach()*angular*sample[:,None,None]*(1+2*time**2)[:,None,None]/prediction.numel())

    def test_default_loss_unchanged_and_invalid_weights_rejected(self):
        p=torch.randn(3,8,3);t=torch.randn_like(p)
        self.assertTrue(torch.equal(velocity_loss(p,t),(p-t).square().mean()))
        with self.assertRaises(ValueError):
            velocity_loss(p,t,sample_weights=torch.tensor([1.,-1.,1.]))


if __name__=='__main__':unittest.main()
