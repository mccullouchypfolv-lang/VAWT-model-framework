"""Train-only periodic Gaussian sources on theta in [0, 2*pi).

The finite-grid spectral admissibility estimate and Gaussian linear-bridge
diagnostics are empirical proxies, not guarantees for a conditional flow model.
All spectra use the forward-normalized Fourier covariance convention.
"""
import numpy as np


class AdmissibilityError(ValueError):
    """The conservative tail estimate permits no positive Matérn smoothness."""


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _validated_psd(spec):
    try:
        n = _positive_integer(spec['n_theta'], 'n_theta')
        psd = np.asarray(spec['psd'], dtype=float)
    except (KeyError, TypeError) as exc:
        raise ValueError('spec requires n_theta and psd') from exc
    if n < 2 or spec.get('n_channels', 1) != 1:
        raise ValueError('source requires n_theta >= 2 and one channel')
    if psd.shape != (n // 2 + 1,) or not np.isfinite(psd).all() or np.any(psd < 0):
        raise ValueError('psd must be a finite nonnegative rFFT spectrum')
    return n, psd


def _multiplicity(n):
    weights = np.full(n // 2 + 1, 2.)
    weights[0] = 1.
    if n % 2 == 0:
        weights[-1] = 1.
    return weights


def _binned_tail(psd, band, width, threshold):
    lo, hi = band
    bins = []
    for start in range(lo, hi + 1, width):
        stop = min(start + width, hi + 1)
        # Include an incomplete terminal bin, without overweighting silent bins.
        power = float(np.mean(psd[start:stop]))
        if power > threshold:
            bins.append((float(np.mean(np.arange(start, stop))), power))
    if len(bins) < 3:
        raise ValueError('tail has fewer than three frequency bins above numerical noise')
    xy = np.log(np.asarray(bins))
    slope, intercept = np.polyfit(xy[:, 0], xy[:, 1], 1)
    error = np.sum((xy[:, 1] - (slope * xy[:, 0] + intercept)) ** 2)
    spread = np.sum((xy[:, 1] - xy[:, 1].mean()) ** 2)
    r2 = 1. - error / spread if spread > 0 else 0.
    return float(-slope), float(r2), bins


def fit_source(train_samples, *, tail_band=(12, 60), resolved_modes=40,
               bin_width=3, bootstrap=200, confidence=.95,
               admissibility_margin=.05, seed=42):
    """Fit a zero-mean Matérn covariance using only the supplied [M,N,1] array.

    Population covariance (ddof=0) describes the empirical training measure.
    Case bootstrap resamples whole curves and recomputes their centered PSD.
    The source retains train DC variance, but does not reproduce the train mean.
    """
    if np.iscomplexobj(train_samples):
        raise ValueError('training samples must be real periodic curves')
    samples = np.asarray(train_samples, dtype=float)
    if samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1] < 4 or samples.shape[2] != 1:
        raise ValueError('train_samples must have shape [M>=2,N>=4,1]')
    if not np.isfinite(samples).all():
        raise ValueError('training samples must be finite')
    n = samples.shape[1]
    width = _positive_integer(bin_width, 'bin_width')
    reps = _positive_integer(bootstrap, 'bootstrap')
    resolved = _positive_integer(resolved_modes, 'resolved_modes')
    if resolved > n // 2:
        raise ValueError('resolved_modes exceeds available rFFT modes')
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be between zero and one')
    if not np.isfinite(admissibility_margin) or admissibility_margin < 0:
        raise ValueError('admissibility_margin must be finite and nonnegative')
    if len(tail_band) != 2:
        raise ValueError('tail_band requires two integer frequency bounds')
    lo, hi = (_positive_integer(v, 'tail_band') for v in tail_band)
    if lo >= hi or hi > n // 2:
        raise ValueError('tail_band must lie within positive rFFT modes')

    z = np.fft.rfft(samples[..., 0], axis=1, norm='forward')
    centered = z - z.mean(axis=0)
    target = np.mean(abs(centered) ** 2, axis=0)
    power = np.mean(abs(z) ** 2, axis=0)
    threshold = float(max(target.max() * np.finfo(float).eps * n, np.finfo(float).tiny))
    beta, r2, bins = _binned_tail(target, (lo, hi), width, threshold)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(reps):
        zb = z[rng.integers(0, len(z), size=len(z))]
        tb = np.mean(abs(zb - zb.mean(axis=0)) ** 2, axis=0)
        slopes.append(_binned_tail(tb, (lo, hi), width, threshold)[0])
    alpha = (1 - confidence) / 2
    ci = np.quantile(slopes, [alpha, 1-alpha])
    upper = float(ci[0] / 2 - 1 - admissibility_margin)
    if upper <= 0:
        raise AdmissibilityError(f'No positive nu satisfies nu < {upper:.6g}; beta CI={ci.tolist()}')

    k = np.arange(n // 2 + 1, dtype=float)
    floor = float(max(target.max() * 1e-12, np.finfo(float).tiny))
    safe_target = np.maximum(target, floor)
    fit_k = k[1:resolved+1]
    log_target = np.log(safe_target[1:resolved+1])
    best = None
    for nu in np.linspace(upper / 50, upper * (1-1e-6), 50):
        for kappa in np.geomspace(.3, 30., 60):
            log_shape = -(nu + .5) * np.log(kappa*kappa + fit_k*fit_k)
            log_a = float(np.mean(log_target - log_shape))
            loss = float(np.mean((log_a + log_shape - log_target)**2))
            if best is None or loss < best[0]:
                best = (loss, float(nu), float(kappa), log_a)
    loss, nu, kappa, log_a = best
    amplitude = float(np.exp(log_a))
    psd = amplitude * (kappa*kappa + k*k) ** -(nu+.5)
    psd[0] = target[0]
    weights = _multiplicity(n)
    mu = safe_target / np.maximum(psd, floor)
    bridge = .5 * (np.sqrt(mu) + 1 / np.sqrt(mu))
    cov = centered.conj().T @ centered / len(z)
    energy = float(np.sum(abs(cov)**2))
    offdiag = float(max(0., energy - np.sum(abs(np.diag(cov))**2)))
    sensitivity = []
    for band in ((max(1, lo-width), hi), (lo+width, hi), (lo, max(lo+1, hi-width))):
        try:
            b, rs, _ = _binned_tail(target, band, width, threshold)
            sensitivity.append({'tail_band': list(band), 'beta': b, 'r_squared': rs})
        except ValueError:
            pass
    return {
        'kind': 'spectral_matern', 'n_theta': n, 'n_channels': 1,
        'domain': [0., float(2*np.pi)], 'fft_norm': 'forward',
        'beta': beta, 'beta_ci': ci.tolist(), 'beta_r_squared': r2,
        'nu_upper': upper, 'nu': nu, 'kappa': kappa, 'amplitude': amplitude,
        'psd': psd.tolist(), 'target_psd': target.tolist(), 'power_psd': power.tolist(),
        'tail_band': [lo, hi], 'resolved_modes': resolved,
        'diagnostics': {
            'training_cases': len(samples), 'bootstrap': reps, 'confidence': confidence,
            'admissibility_margin': admissibility_margin, 'bin_width': width,
            'tail_bins': bins, 'tail_band_sensitivity': sensitivity,
            'noise_threshold': threshold, 'target_psd_floor': floor,
            'floored_modes': np.flatnonzero(target < floor).tolist(),
            'log_spectral_mse': loss, 'mu': mu.tolist(), 'linear_bridge_L': bridge.tolist(),
            'covariance_offdiagonal_energy_fraction': offdiag / energy if energy else 0.,
            'source_total_variance': float(weights @ psd),
            'target_total_variance': float(weights @ target),
            'total_variance_difference': float(weights @ (psd-target)),
            'scope': 'Finite-grid empirical admissibility and Gaussian linear-bridge proxy; no actual conditional-FM guarantee.',
            'labels_augmented': False,
        },
    }


def sample_source(spec, count, seed=None, rng=None):
    """Draw zero-mean real periodic fields [count,N,1], without per-draw scaling."""
    if spec.get('kind')=='floral_fixed_matern_gp':
        from .floral_source import sample_floral_source
        return sample_floral_source(spec,count,seed=seed,rng=rng)
    n, psd = _validated_psd(spec)
    count = _positive_integer(count, 'count')
    if seed is not None and rng is not None:
        raise ValueError('pass either seed or rng, not both')
    if rng is None:
        rng = np.random.default_rng(seed)
    elif not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy Generator')
    z = (rng.normal(size=(count, len(psd))) +
         1j * rng.normal(size=(count, len(psd)))) * np.sqrt(psd / 2)
    z[:, 0] = rng.normal(size=count) * np.sqrt(psd[0])
    if n % 2 == 0:
        z[:, -1] = rng.normal(size=count) * np.sqrt(psd[-1])
    return (np.fft.irfft(z, n=n, axis=1) * n)[..., None]


def _control(spec, shape, kind, **params):
    n, source_psd = _validated_psd(spec)
    weights = _multiplicity(n)
    total = float(weights @ source_psd)
    scale = total / float(weights @ shape)
    psd = shape * scale
    target_total = float(weights @ np.asarray(spec.get('target_psd', source_psd)))
    return {'kind': kind, 'n_theta': n, 'n_channels': 1, 'fft_norm': 'forward',
            'psd': psd.tolist(), 'amplitude': scale, **params,
            'diagnostics': {'source_total_variance': total, 'target_total_variance': target_total,
                            'total_variance_difference': total-target_total,
                            'labels_augmented': False}}


def white_source(spec):
    """Discrete white-noise control with the selected source's total variance."""
    n, _ = _validated_psd(spec)
    return _control(spec, np.ones(n//2+1), 'white')


def fixed_matern_source(spec, nu=.5, kappa=3.):
    """Fixed Matérn control, normalized to the selected source's total variance."""
    n, _ = _validated_psd(spec)
    if not np.isfinite(nu) or not np.isfinite(kappa) or nu <= 0 or kappa <= 0:
        raise ValueError('nu and kappa must be positive and finite')
    k = np.arange(n//2+1, dtype=float)
    return _control(spec, (kappa*kappa+k*k)**-(nu+.5), 'fixed_matern', nu=float(nu), kappa=float(kappa))
