"""Evaluate fixed periodic input functions and the training-resolved source.

These helpers do not resample labels or predictions. Source evaluation keeps the
original fitted spectrum and random draw, with no added modes, refitting, or
resolution-dependent normalization. It defines a bandlimited resolved process,
not the full infinite-frequency Matérn process.
"""
import numpy as np

from .spectral_source import sample_source


SOURCE_SCOPE = 'training-resolved periodic Matérn process'
_PERIOD = 2 * np.pi


def source_scope(source):
    channels=source.get('channels',[])
    if channels and all(c.get('kind')=='floral_fixed_matern_gp' for c in channels):
        from .floral_source import SCOPE
        return SCOPE
    return SOURCE_SCOPE


def midpoint_theta(n_theta):
    """Return N ordered cell midpoints on [0, 2*pi), for integer N >= 4."""
    if (isinstance(n_theta, (bool, np.bool_))
            or not isinstance(n_theta, (int, np.integer)) or n_theta < 4):
        raise ValueError('n_theta must be an integer >= 4')
    return (np.arange(int(n_theta), dtype=float) + .5) * (_PERIOD / int(n_theta))


def _uniform_theta(theta, name):
    if np.iscomplexobj(theta):
        raise ValueError(f'{name} must be real')
    original = np.asarray(theta)
    grid = np.asarray(theta, dtype=float)
    if (grid.ndim != 1 or len(grid) < 4 or not np.isfinite(grid).all()
            or np.any(grid < 0) or np.any(grid >= _PERIOD)):
        raise ValueError(f'{name} must be a finite uniform periodic grid in [0, 2*pi)')
    spacing = _PERIOD / len(grid)
    # Cached angles are float32. Recognize their uniform grid within storage
    # precision, then use its exact spacing rather than fitting rounded nodes.
    precision = np.finfo(original.dtype).eps if original.dtype.kind == 'f' else np.finfo(float).eps
    # JSON/checkpoint .tolist() turns float32 nodes into Python floats and loses
    # the dtype. Recover only exact float32 roundtrips, not arbitrary grids.
    if np.array_equal(grid, grid.astype(np.float32).astype(float)):
        precision = max(precision, np.finfo(np.float32).eps)
    tolerance = max(1e-12, 2 * _PERIOD * precision)
    if not np.allclose(np.diff(np.r_[grid, grid[0] + _PERIOD]), spacing,
                       rtol=1e-9, atol=tolerance) or np.any(np.diff(grid) <= 0):
        raise ValueError(f'{name} must contain one ordered uniform periodic grid')
    if abs(grid[0] - spacing/2) <= tolerance:
        return midpoint_theta(len(grid))
    return grid[0] + np.arange(len(grid)) * spacing


def evaluate_periodic_profile(values, reference_theta, theta):
    """Evaluate a real resolved profile [N] or [N,C] at a uniform query grid.

    Fourier phases refer to the first reference node, so midpoint and other
    uniform-grid offsets are respected. For even N, the real Nyquist coefficient
    continues as a cosine about that node (equivalently, equal half coefficients
    at positive and negative Nyquist frequency). The reference grid is exact.
    """
    reference = _uniform_theta(reference_theta, 'reference_theta')
    query = _uniform_theta(theta, 'theta')
    if np.iscomplexobj(values):
        raise ValueError('values must be real')
    profile = np.asarray(values, dtype=float)
    if (profile.ndim not in (1, 2) or profile.shape[0] != len(reference)
            or not np.isfinite(profile).all()):
        raise ValueError('values must be finite with shape [N] or [N,C]')
    if np.array_equal(reference, query):
        return profile.copy()
    coefficients = np.fft.rfft(profile, axis=0, norm='forward')
    modes = np.arange(len(coefficients), dtype=float)
    weights = np.full(len(coefficients), 2.)
    weights[0] = 1.
    if len(reference) % 2 == 0:
        weights[-1] = 1.
    basis = np.exp(1j * np.outer(query - reference[0], modes)) * weights
    return (basis @ coefficients).real


def evaluate_source(spec, count, theta, rng):
    """Draw the unchanged resolved source and evaluate it as [count,len(theta),1].

    The source's reference origin is pi/N, matching the training midpoint grid.
    The legacy sampler supplies exactly the same random stream at all query
    resolutions. At native nodes its output is returned unchanged. At other
    nodes the same finite spectral realization is evaluated as a function.

    For even training N the Nyquist term has covariance
    psd[N/2]*cos(N/2*(theta-pi/N))*cos(N/2*(phi-pi/N)); no independent sine
    mode is introduced. Consequently its pointwise variance away from native
    nodes varies with that cosine. It is deliberately never renormalized.
    """
    query = _uniform_theta(theta, 'theta')
    if spec.get('kind')=='floral_fixed_matern_gp':
        from .floral_source import sample_floral_source
        return sample_floral_source(spec,count,theta=query,rng=rng)
    domain = np.asarray(spec.get('domain', [0., _PERIOD]), dtype=float)
    if domain.shape != (2,) or not np.array_equal(domain, [0., _PERIOD]):
        raise ValueError('source domain must be [0, 2*pi]')
    reference = midpoint_theta(spec.get('n_theta'))
    draws = sample_source(spec, count, rng=rng)
    if np.array_equal(reference, query):
        return draws
    values = evaluate_periodic_profile(draws[..., 0].T, reference, query)
    return values.T[..., None]
