import unittest
from unittest.mock import patch
import numpy as np
from lf_operator import cross_validation as cv


class CrossValidationTests(unittest.TestCase):
    def test_nested_folds_are_group_disjoint_reproducible_and_cover_all_cases(self):
        groups=np.array([f'g{i//2}' for i in range(100)])
        tsr=np.tile([1.5,2.,2.5,3.,4.,5.,6.,3.5,4.5,5.5],10)
        folds=cv.grouped_cv_splits(groups,tsr,k=5,seed=42)
        self.assertEqual(folds,cv.grouped_cv_splits(groups,tsr,k=5,seed=42))
        tests=[]
        for split in folds:
            self.assertEqual(set.union(*(set(split[s]) for s in ('train','val','test'))),set(range(100)))
            seen=[set(groups[split[s]]) for s in ('train','val','test')]
            for i in range(3):
                self.assertTrue(seen[i])
                for j in range(i):self.assertFalse(seen[i]&seen[j])
            tests.extend(split['test'])
        self.assertEqual(sorted(tests),list(range(100)))
        self.assertTrue(all(16<=len(s['test'])<=24 for s in folds))

    def test_fold_source_and_normalization_use_training_cases_only(self):
        from lf_operator.blade_pipeline import encode_residual
        rng=np.random.default_rng(2);residual=rng.normal(size=(8,16,3))
        arrays={'residual':residual};split=dict(train=[0,1,2,3],val=[4,5],test=[6,7]);calls=[]
        def fake_fit(x,**kwargs):calls.append(x.copy());return {'psd':[1.]*9}
        cfg={'seed':42,'source_fit':{}}
        with patch('lf_operator.spectral_source.fit_source',side_effect=fake_fit):
            stats,source=cv.fit_fold_source(arrays,split,cfg)
        encoded=encode_residual(residual[split['train']],stats)
        self.assertEqual(len(calls),3)
        for c in range(3):np.testing.assert_array_equal(calls[c],encoded[:,:,c:c+1])
        residual[4:]+=10000
        with patch('lf_operator.spectral_source.fit_source',side_effect=fake_fit):
            stats2,source2=cv.fit_fold_source(arrays,split,cfg)
        self.assertEqual(stats,stats2);self.assertEqual(source,source2)

    def test_robust_summary_keeps_outlier_and_reports_its_effect(self):
        target=np.zeros((5,8,3));prediction=np.ones_like(target);prediction[-1]*=100
        summary=cv.error_summary(prediction,target)
        np.testing.assert_allclose(summary['per_case_rmse_median'],[1.]*3)
        np.testing.assert_allclose(summary['per_case_rmse_max'],[100.]*3)
        np.testing.assert_allclose(summary['pooled_rmse'],[np.sqrt(2000.8)]*3)

    def test_outer_fold_count_requires_enough_groups(self):
        with self.assertRaises(ValueError):cv.grouped_cv_splits(np.array(['a']*5),np.ones(5),k=5,seed=42)

    def test_oof_alignment_rejects_missing_or_duplicate_predictions(self):
        values=np.arange(4*8*3).reshape(4,8,3)
        a,fold=cv.assemble_oof(4,[(1,[2,0],values[[2,0]]),(2,[3,1],values[[3,1]])],(8,3))
        np.testing.assert_array_equal(a,values)
        np.testing.assert_array_equal(fold,[1,2,1,2])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            cv.assemble_oof(4,[(1,[0,1],values[[0,1]]),(2,[1,2],values[[1,2]])],(8,3))
        with self.assertRaisesRegex(ValueError,'missing'):
            cv.assemble_oof(4,[(1,[0,1],values[[0,1]])],(8,3))

    def test_fold_subprocess_runs_stage3_with_exact_config_and_checks_failure(self):
        from pathlib import Path
        with patch.object(cv.subprocess,'run') as run:
            cv.train_fold_isolated(Path('configs/example.json'))
        args=run.call_args.args[0]
        self.assertEqual(args[0],cv.sys.executable)
        self.assertEqual(args[-3:],['stage3','--config',str(Path('configs/example.json').resolve())])
        self.assertTrue(run.call_args.kwargs['check'])


if __name__=='__main__':unittest.main()
