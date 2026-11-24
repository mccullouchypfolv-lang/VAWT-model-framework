import unittest
import torch
from lf_operator.residual_flow import ConditionalVelocity,VelocityBlock,apply_film


class ChannelMLPAblationTests(unittest.TestCase):
    def model(self,enabled):
        return ConditionalVelocity(width=8,blocks=2,modes=3,height_width=2,
            direct_conditioning=True,channel_mlp_skip=enabled)

    def test_only_new_mlp_and_gate_parameters_differ_at_same_seed(self):
        torch.manual_seed(42);baseline=self.model(False)
        torch.manual_seed(42);modified=self.model(True)
        old=baseline.state_dict();new=modified.state_dict()
        for name,value in old.items():torch.testing.assert_close(value,new[name],rtol=0,atol=0)
        additions=set(new)-set(old)
        self.assertTrue(additions)
        self.assertTrue(all('.channel_mlp.' in k or k.endswith('.channel_gate') for k in additions))
        self.assertTrue(all(isinstance(block.norm,torch.nn.GroupNorm) for block in modified.blocks))
        self.assertEqual(modified.config['time_embedding_dim'],32)
        restored=ConditionalVelocity(**modified.config);restored.load_state_dict(new,strict=True)

    def test_softgate_bypasses_from_original_block_input_before_film_and_norm(self):
        block=VelocityBlock(4,2,3);block.add_channel_mlp()
        with torch.no_grad():
            for p in block.parameters():p.zero_()
            block.channel_gate.fill_(2.)
            block.channel_mlp[-1].bias.copy_(torch.tensor([0.,.1,.2,.3]))
            block.norm.weight.fill_(1.)
        x=torch.arange(16,dtype=torch.float32).reshape(1,4,4)/10
        modulation=torch.zeros(1,3,4)
        expected=block.norm(apply_film(2*x+torch.tensor([0.,.1,.2,.3])[None,:,None],block.film(modulation)))
        torch.testing.assert_close(block(x,modulation),expected)

    def test_gradients_and_cross_resolution(self):
        model=self.model(True)
        for n in [16,32]:
            condition=model.prepare_condition(torch.randn(2,n,3),torch.randn(2,32,n),torch.randn(2,32,20,n),
                torch.randn(2,2),torch.rand(2,20),torch.ones(2,20),torch.arange(n)[None].expand(2,-1)*2*torch.pi/n)
            out=model(torch.randn(2,n,3),torch.rand(2),condition)
            self.assertEqual(out.shape,(2,n,3));out.square().mean().backward()
            self.assertTrue(all(b.channel_mlp[0].weight.grad is not None and b.channel_gate.grad is not None for b in model.blocks))


if __name__=='__main__':unittest.main()
