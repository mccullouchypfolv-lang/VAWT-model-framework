import unittest
import numpy as np
from lf_operator.spectral_whitening import fit_whitening,whiten,recolor


class SpectralWhiteningTests(unittest.TestCase):
    def test_modal_variance_train_only_and_invertibility(self):
        rng=np.random.default_rng(42);n=32
        q=rng.normal(size=(128,n//2+1,3))+1j*rng.normal(size=(128,n//2+1,3))
        q[:,0]=q[:,0].real;q[:,-1]=q[:,-1].real
        q/=np.arange(1,n//2+2)[None,:,None]**2
        r=np.fft.irfft(q,n=n,axis=1,norm='ortho')
        stats=fit_whitening(r,list(range(100)),variance_floor_ratio=1e-8)
        z=whiten(r,stats);restored=recolor(z,stats)
        np.testing.assert_allclose(restored,r,rtol=1e-5,atol=1e-7)
        fft=np.fft.rfft(z[:100],axis=1,norm='ortho')
        np.testing.assert_allclose(np.mean(fft,axis=0),0.,atol=1e-6)
        np.testing.assert_allclose(np.mean(abs(fft)**2,axis=0),1.,rtol=1e-5)
        changed=r.copy();changed[100:]=1e6
        other=fit_whitening(changed,list(range(100)),variance_floor_ratio=1e-8)
        self.assertEqual(stats,other)
        # Multiple draws/batch axes decode independently.
        np.testing.assert_allclose(recolor(z[:4,None],stats)[:,0],r[:4],rtol=1e-5,atol=1e-7)

    def test_midpoint_resolution_transfer_and_silent_mode_floor(self):
        from lf_operator.resolution import evaluate_periodic_profile,midpoint_theta
        rng=np.random.default_rng(8);n=16;ref=midpoint_theta(n);query=midpoint_theta(32)
        r=np.stack([rng.normal(size=(1,3))+rng.normal(size=(1,3))*np.sin(2*ref[:,None]) for _ in range(20)])
        stats=fit_whitening(r,list(range(16)))
        self.assertTrue(np.isfinite(whiten(r,stats)).all())
        z=whiten(r,stats)
        high=np.stack([evaluate_periodic_profile(case,ref,query) for case in z])
        new=dict(stats,mean_profile=evaluate_periodic_profile(np.array(stats['mean_profile']),ref,query).tolist())
        output=recolor(high,new)
        expected=np.stack([evaluate_periodic_profile(case,ref,query) for case in r])
        np.testing.assert_allclose(output,expected,rtol=1e-5,atol=1e-6)

    def test_existing_pipeline_dispatch_and_white_source_variance(self):
        from lf_operator.blade_pipeline import encode_residual,decode_residual,sample_multichannel_source
        from lf_operator.spectral_whitening import white_reference
        rng=np.random.default_rng(2);r=rng.normal(size=(100,32,3));stats=fit_whitening(r,list(range(80)))
        z=encode_residual(r,stats)
        np.testing.assert_allclose(decode_residual(z,stats),r,rtol=1e-5,atol=1e-6)
        source=white_reference(stats,dict(targets=stats['targets'],fit_indices=list(range(80))))
        g=sample_multichannel_source(source,20000,rng)
        p=np.mean(abs(np.fft.rfft(g,axis=1,norm='ortho'))**2,axis=0)
        np.testing.assert_allclose(p,1.,rtol=.04)


if __name__=='__main__':unittest.main()
