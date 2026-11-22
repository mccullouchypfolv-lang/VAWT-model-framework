import unittest
import numpy as np
from lf_operator import splitting


class LatinSplitTests(unittest.TestCase):
    def test_lf_latin_training_uses_train_only_statistics_with_shared_geometries(self):
        import contextlib,io,json,shutil,uuid,torch
        from pathlib import Path
        from unittest.mock import patch
        from lf_operator import blade_pipeline as bp
        from lf_operator.blade_data import INPUT_CONTRACT,TARGET_CONTRACT,FEATURES,CONDITIONS
        root=Path(__file__).resolve().parents[1]/'tmp'
        folder=root/('lf-lhs-training-'+uuid.uuid4().hex);cache=folder/'work/lf_cache';cache.mkdir(parents=True)
        try:
            rng=np.random.default_rng(9);records=[];ys=[]
            for i in range(6):
                file=f'case{i}.npz';ys.append(rng.normal(size=(16,3)).astype('float32'))
                np.savez(cache/file,x=np.broadcast_to(np.array([3.2,.16],dtype='float32')[None,:,None],(20,2,16)),
                    a=np.array([8+i,2+.5*i],dtype='float32'),y=ys[-1],weights=np.ones(20,dtype='float32'),
                    theta=((np.arange(16)+.5)*2*np.pi/16).astype('float32'),z=(59.6+(np.arange(20)+.5)*112/20).astype('float32'))
                records.append(dict(case_id=str(i),geometry_id='shared',file=file,cache_hash=bp.digest(cache/file)))
            manifest=dict(input_contract=INPUT_CONTRACT,target_contract=TARGET_CONTRACT,features=list(FEATURES),
                conditions=list(CONDITIONS),targets=bp.TARGETS,records=records,
                config=dict(rotor_bottom_height=59.6,height=112,nz=20,ntheta=16))
            (cache/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            cfg=dict(work='work',seed=42,device='cpu',lf_split=dict(method='latin_hypercube'),
                lf_model=dict(width=4,blocks=1,modes_z=2,modes_theta=3,pad_z=2),lf_lr=.001,lf_epochs=1,lf_patience=15,batch_size=2)
            with patch.object(bp,'ROOT',folder),contextlib.redirect_stdout(io.StringIO()):bp.train_lf(cfg)
            ck=torch.load(folder/'work/stage1/best.pt',weights_only=False)
            self.assertEqual(ck['statistics']['fit_indices'],ck['splits']['train'])
            expected=np.concatenate([ys[i] for i in ck['splits']['train']]).mean(0)
            np.testing.assert_allclose(ck['statistics']['y_mean'],expected,atol=1e-7)
            self.assertEqual(json.loads((folder/'work/stage1/split_design.json').read_text())['method'],'latin_hypercube')
            self.assertTrue((folder/'work/stage1/metrics.json').exists())
        finally:
            self.assertTrue(folder.resolve().is_relative_to(root.resolve()));shutil.rmtree(folder)

    def test_sparse_matching_returns_unique_cases_without_dense_cost_matrix(self):
        from unittest.mock import patch
        x=np.random.default_rng(3).uniform(size=(80,4))
        with patch.object(splitting,'DENSE_CASE_LIMIT',20):
            parts,report=splitting.latin_hypercube_split(x,seed=42)
        self.assertEqual(sorted(sum(parts.values(),[])),list(range(80)))
        self.assertEqual(report['assignment_solver'],'sparse_candidate_min_cost')

    def test_parameter_reader_does_not_require_labels(self):
        import shutil,uuid
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]/'tmp'
        folder=root/('split-inputs-'+uuid.uuid4().hex);folder.mkdir(parents=True)
        try:
            x=np.broadcast_to(np.array([3.2,.16])[None,:,None],(20,2,16))
            np.savez(folder/'case.npz',x=x,a=np.array([12.,3.]))
            values=splitting.case_parameter_inputs(folder,[{'file':'case.npz'}])
            np.testing.assert_allclose(values,[[12,3,3.2,.16]])
        finally:
            self.assertTrue(folder.resolve().is_relative_to(root.resolve()));shutil.rmtree(folder)

    def test_lhs_assignment_is_disjoint_complete_and_reproducible(self):
        x=np.random.default_rng(5).uniform(size=(49,4))
        splits,report=splitting.latin_hypercube_split(x,seed=42)
        other,_=splitting.latin_hypercube_split(x,seed=42)
        self.assertEqual(splits,other)
        self.assertEqual({k:len(v) for k,v in splits.items()},dict(train=35,val=7,test=7))
        ids=sum(splits.values(),[])
        self.assertEqual(sorted(ids),list(range(49)))
        for name,design in report['unit_design'].items():
            design=np.asarray(design);n=len(design)
            for d in range(4):self.assertEqual(sorted((design[:,d]*n).astype(int).tolist()),list(range(n)))
        # Changing physical units must not change assignments.
        scaled,_=splitting.latin_hypercube_split(x*[10,2,100,.01]+[5,1,2,.1],seed=42)
        self.assertEqual(splits,scaled)

    def test_matching_can_recover_an_exact_latin_design(self):
        from scipy.stats import qmc
        # Only sample allocation may vary: all rows still match a unique target.
        x=qmc.LatinHypercube(d=4,seed=4).random(30)
        splits,report=splitting.latin_hypercube_split(x,seed=7)
        self.assertTrue(np.isfinite(report['mean_squared_assignment_distance']))
        self.assertEqual(len(set(sum(splits.values(),[]))),30)

    def test_constant_dimensions_and_invalid_inputs(self):
        x=np.ones((12,4));x[:,0]=np.arange(12)
        splits,_=splitting.latin_hypercube_split(x,seed=42)
        self.assertEqual(len(sum(splits.values(),[])),12)
        for bad in [np.ones(12),np.ones((2,4)),np.full((12,4),np.nan)]:
            with self.assertRaises(ValueError):splitting.latin_hypercube_split(bad,seed=42)

    def test_geometry_overlap_requires_explicit_matching_source_policy(self):
        from lf_operator.blade_pipeline import validate_flow_assets,residual_normalization,TARGETS
        from lf_operator.blade_data import TARGET_CONTRACT,INPUT_CONTRACT
        residual=np.random.default_rng(2).normal(size=(6,16,3))
        arrays=dict(case_ids=np.array(list('abcdef')),geometry_ids=np.array(['same']*6),
            target=residual,lf_prediction=np.zeros_like(residual),residual=residual.copy())
        splits=dict(train=[0,1,2,3],val=[4],test=[5])
        stats=residual_normalization(residual,splits['train'])
        source=dict(channels=[{}]*3,targets=TARGETS,fit_indices=splits['train'],
            target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT)
        with self.assertRaises(ValueError):validate_flow_assets(arrays,splits,stats,source)
        with self.assertRaises(ValueError):validate_flow_assets(arrays,splits,stats,source,split_method='latin_hypercube')
        source['split_method']='latin_hypercube'
        validate_flow_assets(arrays,splits,stats,source,split_method='latin_hypercube')
        with self.assertRaises(ValueError):validate_flow_assets(arrays,dict(splits,test=[4]),stats,source,split_method='latin_hypercube')


if __name__=='__main__':unittest.main()
