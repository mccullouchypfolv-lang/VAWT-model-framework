import json,math,unittest
import numpy as np
import torch
from lf_operator.residual_flow import ConditionalVelocity
from lf_operator import conditioning


class PhysicalConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(1)

    def model(self):
        return ConditionalVelocity(width=8,blocks=1,modes=5,direct_conditioning=True,
                                   explicit_physical_conditioning=True)

    def inputs(self,n=32):
        return [torch.randn(2,n,3),torch.randn(2,32,n),torch.randn(2,32,20,n),
            torch.randn(2,2,requires_grad=True),torch.linspace(0,1,20).expand(2,-1),torch.ones(2,20),
            (torch.arange(n)*2*math.pi/n).expand(2,-1)]

    def test_geometry_and_operating_parameters_reach_concat_and_film(self):
        m=self.model()
        for route in ['concat','film']:
            args=self.inputs();geom=torch.randn(2,20,2,requires_grad=True)
            c=m.prepare_condition(*args,geometry=geom)
            if route=='concat':c['encoded_features']=c['encoded_features'].detach()
            else:c['features']=c['features'].detach()
            g=torch.randn(2,32,3);t=torch.tensor([.2,.7]);v=m(g,t,c)
            ga,gg=torch.autograd.grad(v.square().mean(),(args[3],geom))
            self.assertGreater(ga.abs().sum().item(),0)
            self.assertGreater(gg.abs().sum().item(),0)
            self.assertTrue(torch.isfinite(gg).all())
            self.assertEqual(c['features'].shape,(2,55,32))
        self.assertEqual(m.lift[0].in_channels,90)
        self.assertEqual(m.condition_encoder[0].in_channels,55)

    def test_checkpoint_roundtrip_and_two_angular_resolutions(self):
        m=self.model()
        clone=ConditionalVelocity(**json.loads(json.dumps(m.config)))
        clone.load_state_dict(m.state_dict(),strict=True)
        for n in [32,64]:
            args=self.inputs(n);geometry=torch.randn(2,20,2)
            c=m.prepare_condition(*args,geometry=geometry)
            g=torch.randn(2,n,3);t=torch.tensor([.2,.7])
            torch.testing.assert_close(m(g,t,c),clone(g,t,c))
        with self.assertRaises(ValueError):m.prepare_condition(*self.inputs())
        with self.assertRaises(ValueError):ConditionalVelocity(explicit_physical_conditioning=True)

    def test_v1_weights_and_common_initialization_are_preserved(self):
        torch.manual_seed(42);v1=ConditionalVelocity(width=8,blocks=1,modes=5,direct_conditioning=True)
        torch.manual_seed(42);v2=self.model()
        for name,p in v1.state_dict().items():
            q=v2.state_dict()[name]
            if name in ['lift.0.weight','condition_encoder.0.weight']:q=q[:,:p.shape[1]]
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        config={k:v for k,v in v1.config.items() if k not in ['explicit_physical_conditioning','geometry_width']}
        old=ConditionalVelocity(**config);old.load_state_dict(v1.state_dict(),strict=True)

    def test_geometry_scalers_use_training_cases_only(self):
        rng=np.random.default_rng(42)
        a=dict(conditions=rng.normal(size=(4,2)),lf_prediction=rng.normal(size=(4,16,3)),
            rotor_features=rng.normal(size=(4,32,16)),field_features=rng.normal(size=(4,32,20,16)),
            geometry=rng.normal(size=(4,20,2)))
        s=conditioning.fit_condition_scalers(a,[0,1],include_geometry=True)
        a['geometry'][2:]+=10000
        self.assertEqual(s,conditioning.fit_condition_scalers(a,[0,1],include_geometry=True))
        encoded=conditioning.standardize_conditions(a,s)
        np.testing.assert_allclose(encoded['geometry'][:2].mean((0,1)),0,atol=1e-6)
        self.assertNotIn('geometry',conditioning.fit_condition_scalers(a,[0,1]))

    def test_geometry_cache_loader_reads_only_parameter_inputs_and_checks_identity(self):
        import shutil,uuid
        from pathlib import Path
        from lf_operator.data import digest
        root=Path(__file__).resolve().parents[1]/'tmp';folder=root/('physical-cache-'+uuid.uuid4().hex)
        folder.mkdir(parents=True)
        try:
            g=np.stack([np.linspace(3,4,20),np.linspace(.12,.2,20)],axis=-1).astype('float32')
            x=np.repeat(g[:,:,None],16,axis=2)
            np.savez(folder/'a.npz',x=x,a=np.array([12.,3.]))
            record=dict(case_id='a',file='a.npz',cache_hash=digest(folder/'a.npz'))
            arr=dict(case_ids=np.array(['a']),conditions=np.array([[12.,3.]]))
            got=conditioning.geometry_from_cache(arr,folder,dict(records=[record]))
            np.testing.assert_array_equal(got['geometry'][0],g)
            self.assertNotIn('geometry',arr)
            with self.assertRaises(ValueError):conditioning.geometry_from_cache(dict(arr,conditions=np.array([[8.,3.]])),folder,dict(records=[record]))
            record['cache_hash']='wrong'
            with self.assertRaises(ValueError):conditioning.geometry_from_cache(arr,folder,dict(records=[record]))
        finally:
            self.assertTrue(folder.resolve().is_relative_to(root.resolve()));shutil.rmtree(folder)


if __name__=='__main__':unittest.main()
