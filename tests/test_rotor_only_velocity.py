import unittest
import numpy as np
import torch
from lf_operator.rotor_only_velocity import RotorOnlyVelocity
from lf_operator.residual_flow import ConditionalVelocity,create_velocity_model
from lf_operator.conditioning import fit_condition_scalers,standardize_conditions,tensor_conditions,prepare_condition


class RotorOnlyVelocityTests(unittest.TestCase):
    def test_retained_columns_and_shared_weights_match_same_seed_default(self):
        options=dict(width=8,blocks=2,modes=6,lift_ratio=2,projection_ratio=2,direct_conditioning=True)
        torch.manual_seed(42);old=ConditionalVelocity(**options)
        torch.manual_seed(42);new=RotorOnlyVelocity(**options)
        self.assertFalse(hasattr(new,'height_lift'))
        self.assertEqual(new.lift[0].in_channels,72)
        self.assertEqual(new.condition_encoder[0].in_channels,39)
        torch.testing.assert_close(new.lift[0].weight,old.lift[0].weight[:,list(range(70))+[78,79]])
        torch.testing.assert_close(new.condition_encoder[0].weight,old.condition_encoder[0].weight[:,list(range(35))+list(range(43,47))])
        for key,value in new.state_dict().items():
            if key not in ('lift.0.weight','condition_encoder.0.weight'):
                torch.testing.assert_close(value,old.state_dict()[key])
        self.assertLess(sum(p.numel() for p in new.parameters()),sum(p.numel() for p in old.parameters()))

    def test_field_and_height_free_conditioning_and_resolution_roundtrip(self):
        model=RotorOnlyVelocity(width=8,blocks=1,modes=6,lift_ratio=2,projection_ratio=2,direct_conditioning=True)
        for n in (16,32):
            arrays={'case_ids':np.array(['a','b','c']), 'theta':(np.arange(n)+.5)*2*np.pi/n,
                'conditions':np.arange(6).reshape(3,2),'lf_prediction':np.random.default_rng(1).normal(size=(3,n,3)),
                'rotor_features':np.random.default_rng(2).normal(size=(3,32,n))}
            stats=fit_condition_scalers(arrays,[0,1],include_field=False)
            self.assertNotIn('field_features',stats)
            tensors=tensor_conditions(standardize_conditions(arrays,stats),arrays,'cpu',include_height=False)
            self.assertNotIn('z',tensors);self.assertNotIn('weights',tensors)
            condition=prepare_condition(model,tensors,[0,1])
            self.assertEqual(condition['features'].shape,(2,37,n))
            state=torch.randn(2,n,3);time=torch.tensor([.2,.8]);prediction=model(state,time,condition)
            self.assertEqual(prediction.shape,state.shape);prediction.square().mean().backward()
            restored=create_velocity_model(**model.config);restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(prediction,restored(state,time,condition))


if __name__=='__main__':unittest.main()
