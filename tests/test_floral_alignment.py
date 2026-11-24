import unittest
import numpy as np
import torch
from lf_operator import residual_flow
from lf_operator import flow_training


class FloralAlignmentTests(unittest.TestCase):
    def test_block_uses_original_input_skip_then_film_after_mlp(self):
        block=residual_flow.FloralVelocityBlock(2,2,2,last=True)
        with torch.no_grad():
            for p in block.parameters():p.zero_()
            block.channel_gate.fill_(2.)
            block.channel_mlp[-1].bias.fill_(.25)
            block.film[-1].bias[:2].fill_(.5)
            block.film[-1].bias[2:].fill_(.1)
        x=torch.tensor([[[-2.,-1.,1.,2.],[1.,-2.,3.,-4.]]])
        actual=block(x,torch.zeros_like(x))
        torch.testing.assert_close(actual,1.5*(.25+2*x)+.1)

    def test_time_and_grid_follow_official_encoding(self):
        enc=residual_flow.FloralTimeEmbedding()
        t=torch.tensor([0.,.123,1.])
        phase=t[:,None]*2*torch.pi*torch.logspace(-1,2,5)[None]
        torch.testing.assert_close(enc(t),torch.cat((phase.sin(),phase.cos()),1))
        grid=residual_flow.floral_grid(8,device='cpu',dtype=torch.float32)
        torch.testing.assert_close(grid,torch.arange(8,dtype=torch.float32)/8)

    def test_dataloader_visits_every_training_case_exactly_once(self):
        ids=list(range(71));batches=list(flow_training.epoch_indices(ids,batch_size=256,seed=42,epoch=1))
        self.assertEqual(len(batches),1);self.assertEqual(sorted(np.concatenate(batches).tolist()),ids)
        more=list(flow_training.epoch_indices(ids,batch_size=32,seed=42,epoch=1))
        self.assertEqual([len(x) for x in more],[32,32,7])
        np.testing.assert_array_equal(np.concatenate(more),np.concatenate(batches))
        again=np.concatenate(list(flow_training.epoch_indices(ids,batch_size=32,seed=42,epoch=2)))
        self.assertFalse(np.array_equal(again,np.concatenate(more)))

    def test_bridge_matches_official_mse_scale_without_doubling_target(self):
        source=dict(channels=[dict(n_theta=16,n_channels=1,psd=(1/(1+np.arange(9)**2)).tolist())]*3)
        g=np.zeros((4,16,3),dtype='float32');target=np.full_like(g,2);t=np.full((4,1,1),.4,dtype='float32')
        cfg=dict(method='floral',sig_min=.01,prior_covariance_factor=2.)
        noise=flow_training.base_bridge_noise(source,4,np.random.default_rng(7),prior_covariance_factor=2.)
        actual=flow_training.flow_path(g,target,t,source,np.random.default_rng(7),cfg)
        np.testing.assert_allclose(actual,(1-t)*g+t*target+.01*4*noise,atol=1e-7)
        from lf_operator.blade_pipeline import sample_multichannel_source
        prior=sample_multichannel_source(source,4,np.random.default_rng(7))
        np.testing.assert_allclose(noise*np.sqrt(2),prior,atol=2e-7)

    def test_new_architecture_has_channel_mlp_no_groupnorm_and_resolution_support(self):
        model=residual_flow.ConditionalVelocity(width=8,blocks=2,modes=3,height_width=2,
            architecture='floral-filmfno-aligned-v2',time_embedding_dim=10,direct_conditioning=True)
        self.assertFalse(any(isinstance(m,torch.nn.GroupNorm) for m in model.modules()))
        self.assertEqual(len(model.blocks),2)
        self.assertTrue(all(hasattr(block,'channel_mlp') for block in model.blocks))
        for n in [16,32]:
            condition=model.prepare_condition(torch.randn(2,n,3),torch.randn(2,32,n),torch.randn(2,32,20,n),
                torch.randn(2,2),torch.rand(2,20),torch.ones(2,20),torch.arange(n)[None].expand(2,-1)*2*torch.pi/n)
            torch.testing.assert_close(condition['features'][:,-1],residual_flow.floral_grid(n,device='cpu',dtype=torch.float32)[None].expand(2,-1))
            self.assertEqual(model.blocks[0].film[0].in_channels,8)
            self.assertEqual(condition['raw_condition'].shape[1],model.condition_encoder[0].in_channels-1)
            self.assertEqual(model.linear_film[0].in_channels,condition['raw_condition'].shape[1])
            pred=model(torch.randn(2,n,3),torch.rand(2),condition)
            self.assertEqual(pred.shape,(2,n,3))
            pred.square().mean().backward()
            self.assertTrue(all(block.channel_mlp[0].weight.grad is not None for block in model.blocks))
        restored=residual_flow.ConditionalVelocity(**model.config);restored.load_state_dict(model.state_dict())
        self.assertEqual(sum(isinstance(m,torch.nn.Conv1d) for m in model.linear_operator),4)


if __name__=='__main__':unittest.main()
