import unittest
import numpy as np
import torch
from lf_operator import low_frequency_fm as spectral
from lf_operator.objectives import velocity_loss


class LowFrequencyFMTests(unittest.TestCase):
    def test_full_spectrum_matches_pointwise_loss_and_gradients_even_and_odd(self):
        for n in (15,16):
            torch.manual_seed(42)
            p=torch.randn(3,n,3,dtype=torch.float64,requires_grad=True);y=torch.randn_like(p)
            time=torch.tensor([0.,.5,1.],dtype=torch.float64)
            expected=velocity_loss(p,y,time=time,time_weighting='floral_quadratic')
            actual=spectral.low_frequency_velocity_loss(p,y,np.ones((n//2+1,3)),time=time,time_weighting='floral_quadratic')
            torch.testing.assert_close(actual,expected)
            a=torch.autograd.grad(actual,p,retain_graph=True)[0];b=torch.autograd.grad(expected,p)[0]
            torch.testing.assert_close(a,b)

    def test_complex_coefficients_detect_phase_error_and_exclude_unselected_modes(self):
        n=32;theta=torch.arange(n,dtype=torch.float64)*2*torch.pi/n
        p=torch.cos(theta)[None,:,None];y=torch.sin(theta)[None,:,None]
        weights=np.zeros((17,1));weights[1]=1
        self.assertAlmostEqual(float(spectral.low_frequency_velocity_loss(p,y,weights)),1.,places=10)
        high=torch.cos(8*theta)[None,:,None]
        self.assertLess(float(spectral.low_frequency_velocity_loss(high,torch.zeros_like(high),weights)),1e-20)

    def test_fit_is_channelwise_train_only_and_uses_energy_cutoff(self):
        from lf_operator.blade_pipeline import residual_normalization
        n=32;theta=np.arange(n)*2*np.pi/n
        curve=np.stack([np.cos(k*theta)+.02*np.cos(9*theta) for k in (1,2,3)],axis=-1)
        values=np.array([a*curve for a in (-1.,-.5,.5,1.,2.,3.)])
        ids=[0,1,2,3];stats=residual_normalization(values,ids)
        fit=spectral.fit_low_frequency_weights(values,ids,stats,dict(energy_fraction=.95,energy_power=.5,strength=1.))
        self.assertEqual(fit['cutoff_modes'],[1,2,3])
        self.assertEqual(fit['fit_indices'],ids)
        weights=np.asarray(fit['weights']);m=np.full(n//2+1,2.);m[0]=m[-1]=1
        for c,k in enumerate(fit['cutoff_modes']):
            self.assertTrue((weights[k+1:,c]==0).all())
            self.assertAlmostEqual(float(np.sum(weights[:k+1,c]*m[:k+1])/m[:k+1].sum()),1.)
        values[4:]+=10000
        self.assertEqual(fit,spectral.fit_low_frequency_weights(values,ids,stats,dict(energy_fraction=.95,energy_power=.5,strength=1.)))

    def test_invalid_fit_settings_fail(self):
        values=np.ones((4,16,3));stats={'mean_profile':np.zeros((16,3)).tolist(),'scale':[1.,1.,1.],'fit_indices':[0,1]}
        for options in ({'energy_fraction':0},{'energy_fraction':1.1},{'energy_power':-1},{'strength':-1}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                spectral.fit_low_frequency_weights(values,[0,1],stats,options)


if __name__=='__main__':unittest.main()
