"""Full checkpoint/condition/source/ODE pipeline on new angular grids."""
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

from lf_operator.blade_pipeline import ROOT,TARGETS,predict,lf_predict_parameters
from lf_operator.blade_data import INPUT_CONTRACT,TARGET_CONTRACT
from lf_operator.conditioning import fit_condition_scalers
from lf_operator.data import digest
from lf_operator.residual_flow import ConditionalVelocity
from tests import test_parameter_pipeline as parameter_fixture


class ResolutionPredictionTests(unittest.TestCase):
    def test_same_checkpoint_end_to_end_at_four_resolutions(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        lf=parameter_fixture.ParameterPipelineTests().checkpoint()
        queries=[dict(wind_ref_m_s=9.,tsr_ref=4.,chord_m=3.,thickness_ratio=.18),
                 dict(wind_ref_m_s=12.,tsr_ref=5.,chord_m=4.,thickness_ratio=.21)]
        arrays=lf_predict_parameters(lf,queries,'cpu')
        scalers=fit_condition_scalers(arrays,[0,1])
        model=ConditionalVelocity(width=4,blocks=1,modes=3,height_width=2,field_channels=4,rotor_channels=4)
        channel=dict(n_theta=160,n_channels=1,psd=(1/(1+np.arange(81)**2)**2).tolist())
        theta=arrays['theta']
        stats=dict(mean_profile=np.stack([np.sin(theta),np.cos(theta),np.sin(2*theta)],-1).tolist(),scale=[1.,.1,.01])
        (ROOT/'tmp').mkdir(exist_ok=True)
        work=ROOT/'tmp'/('resolution-test-'+uuid.uuid4().hex)
        work.mkdir()
        try:
            (work/'stage1').mkdir();(work/'stage3').mkdir()
            torch.save(lf,work/'stage1/best.pt')
            ck=dict(model=model.state_dict(),model_config=model.config,targets=TARGETS,
                target_contract=TARGET_CONTRACT,input_contract=INPUT_CONTRACT,
                lf_checkpoint_sha256=digest(work/'stage1/best.pt'),theta=theta.tolist(),
                scalers=scalers,source={'channels':[channel]*3},residual_stats=stats)
            torch.save(ck,work/'stage3/best.pt')
            cfg=dict(work=str(work),device='cpu',seed=42,evaluation_draws=2,ode_steps=2)
            for n in [80,160,320,640]:
                output=work/f'prediction_{n}.npz'
                predict(cfg,None,output,parameters=queries,n_theta=n)
                with np.load(output) as result:
                    self.assertEqual(result['samples'].shape,(2,2,n,3))
                    self.assertEqual(result['mean'].shape,(2,n,3))
                    self.assertEqual(len(result['training_theta']),160)
                    self.assertTrue(np.isfinite(result['samples']).all())
                    self.assertTrue(np.all(result['q05']<=result['q95']))
            # Default inference remains native and exactly reproducible.
            predict(cfg,None,work/'default.npz',parameters=queries)
            with np.load(work/'default.npz') as default,np.load(work/'prediction_160.npz') as explicit:
                np.testing.assert_array_equal(default['samples'],explicit['samples'])
        finally:
            if not work.resolve().is_relative_to((ROOT/'tmp').resolve()):
                raise ValueError('test cleanup outside workspace')
            shutil.rmtree(work)


if __name__=='__main__':unittest.main()
