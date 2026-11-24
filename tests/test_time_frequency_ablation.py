import unittest
import torch
from lf_operator.residual_flow import FourierTimeEmbedding,ConditionalVelocity


class TimeFrequencyTests(unittest.TestCase):
    def test_limits_dimensions_and_legacy_default(self):
        embedding=FourierTimeEmbedding(32,max_frequency=10.)
        self.assertEqual(embedding(torch.tensor([0.,.5,1.])).shape,(3,32))
        torch.testing.assert_close(embedding.frequencies[[0,-1]],torch.tensor([2*torch.pi,20*torch.pi]))
        torch.testing.assert_close(FourierTimeEmbedding(32).frequencies,
            2*torch.pi*torch.exp(torch.linspace(0,__import__('math').log(1000),16)),rtol=0,atol=0)
        for bad in [0.,.5,float('nan'),True]:
            with self.assertRaises(ValueError):FourierTimeEmbedding(32,max_frequency=bad)

    def test_only_time_buffer_changes_same_seed_and_checkpoint_reloads(self):
        torch.manual_seed(42);old=ConditionalVelocity(width=8,blocks=1,modes=6,direct_conditioning=True)
        torch.manual_seed(42);new=ConditionalVelocity(width=8,blocks=1,modes=6,direct_conditioning=True,time_max_frequency=10.)
        changed=[]
        for key,value in old.state_dict().items():
            if not torch.equal(value,new.state_dict()[key]):changed.append(key)
        self.assertEqual(changed,['time_embedding.frequencies'])
        restored=ConditionalVelocity(**new.config);restored.load_state_dict(new.state_dict(),strict=True)
        torch.testing.assert_close(restored.time_embedding.frequencies,new.time_embedding.frequencies)


if __name__=='__main__':unittest.main()
