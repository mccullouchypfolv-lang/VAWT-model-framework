"""Empirical kernel transport using NumPy only.

The signature option is an RBF on finite Chen tensor features, NOT the full
signature kernel and carries none of its characteristic-kernel guarantees.
Raw RBF costs lie in [0, 2]. Batch median scaling does not preserve a
data-independent global bound. Sampling the joint plan preserves marginals
in distribution; finite sampled histograms need not equal those marginals.
"""

import math

import numpy as np


class SinkhornConvergenceError(RuntimeError):
    """Raised when the requested transport marginal tolerance is not reached."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"Sinkhorn did not converge: {diagnostics}")


def _curves(value, name):
    raw = np.asarray(value)
    if not np.isrealobj(raw):
        raise ValueError(f'{name} must be real')
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim != 3 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite nonempty [B,N,C]')
    return result


def _signature_features(curves, depth, time_aug):
    """Piecewise-linear signature through depth, with a zero basepoint.

    Time is normalized to [0,1]. The basepoint segment records initial level.
    Tensor coordinates use the ordinary Euclidean inner product.
    """
    if not time_aug and curves.shape[2] == 1:
        # In one dimension every iterated integral depends only on total
        # increment. Use this exact identity to avoid amplifying roundoff
        # on closed paths through the automatic median bandwidth.
        endpoint = curves[:, -1, :]
        return np.concatenate([endpoint**k/math.factorial(k) for k in range(1, depth+1)], axis=1)
    if time_aug:
        time = np.broadcast_to(np.linspace(0., 1., curves.shape[1])[None, :, None],
                               (curves.shape[0], curves.shape[1], 1))
        curves = np.concatenate((time, curves), axis=2)
    dimension = curves.shape[2]
    levels = [np.ones((len(curves), 1))] + [
        np.zeros((len(curves), dimension**k)) for k in range(1, depth + 1)]
    increments = np.diff(np.concatenate((np.zeros_like(curves[:, :1]), curves), axis=1), axis=1)
    for delta in increments.transpose(1, 0, 2):
        segment = [np.ones((len(curves), 1))]
        for k in range(1, depth + 1):
            segment.append((segment[-1][:, :, None] * delta[:, None, :]).reshape(len(curves), -1) / k)
        levels = [sum((levels[k-j][:, :, None] * segment[j][:, None, :]).reshape(len(curves), -1)
                      for j in range(k + 1)) for k in range(depth + 1)]
    return np.concatenate(levels[1:], axis=1)


def _squared_distances(x, y):
    # Direct differences avoid cancellation for almost equal curves.
    result = np.empty((len(x), len(y)), dtype=np.float64)
    for i, row in enumerate(x):
        result[i] = np.sum((y - row)**2, axis=1)
    if not np.isfinite(result).all():
        raise ValueError('kernel feature distances overflow float64; rescale inputs')
    return result


def kernel_cost(source, target, kernel='rbf', bandwidth=None, derivative_weight=.01,
                signature_depth=3, time_aug=True, median_normalize=True):
    """Return squared RKHS costs and JSON-compatible geometric metadata.

    RBF convention: exp(-d_squared/(2*bandwidth**2)); automatic bandwidth
    is sqrt(median(positive d_squared)). RBF L2 uses grid mean and channel sum.
    Sobolev adds derivative_weight times the corresponding derivative norm,
    with periodic centered derivatives on theta in [0, 2*pi).
    """
    x, y = _curves(source, 'source'), _curves(target, 'target')
    if x.shape[1:] != y.shape[1:]:
        raise ValueError('source and target grids/channels must agree')
    if kernel == 'signature':
        kernel = 'signature_truncated_rbf'
    if kernel not in ('rbf', 'sobolev_rbf', 'signature_truncated_rbf'):
        raise ValueError(f'unknown kernel {kernel}')
    if not np.isfinite(derivative_weight) or derivative_weight < 0:
        raise ValueError('derivative_weight must be finite and nonnegative')
    if not isinstance(signature_depth, (int, np.integer)) or not 1 <= signature_depth <= 3:
        raise ValueError('signature_depth must be an integer from 1 to 3')
    if bandwidth is not None and (not np.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('bandwidth must be positive and finite')
    if kernel == 'signature_truncated_rbf':
        fx = _signature_features(x, signature_depth, time_aug)
        fy = _signature_features(y, signature_depth, time_aug)
    else:
        fx, fy = x.reshape(len(x), -1), y.reshape(len(y), -1)
        if kernel == 'sobolev_rbf':
            step = 2*np.pi/x.shape[1]
            dx = (np.roll(x, -1, axis=1)-np.roll(x, 1, axis=1))/(2*step)
            dy = (np.roll(y, -1, axis=1)-np.roll(y, 1, axis=1))/(2*step)
            fx = np.concatenate((fx, np.sqrt(derivative_weight)*dx.reshape(len(x), -1)), axis=1)
            fy = np.concatenate((fy, np.sqrt(derivative_weight)*dy.reshape(len(y), -1)), axis=1)
        fx, fy = fx/np.sqrt(x.shape[1]), fy/np.sqrt(y.shape[1])
    distance = _squared_distances(fx, fy)
    positive = distance[distance > 0]
    actual_bandwidth = (float(np.sqrt(np.median(positive))) if positive.size else 1.) if bandwidth is None else float(bandwidth)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        ratio = (np.sqrt(distance)/actual_bandwidth)**2
    cost = -2*np.expm1(-.5*ratio)
    positive_cost = cost[cost > 0]
    scale = float(np.median(positive_cost)) if median_normalize and positive_cost.size else 1.
    cost = cost/scale
    metadata = dict(kernel=kernel, bandwidth=actual_bandwidth,
                    derivative_weight=float(derivative_weight), signature_depth=int(signature_depth),
                    time_aug=bool(time_aug), median_normalize=bool(median_normalize),
                    cost_median_scale=scale, raw_cost_bound=2.,
                    normalized_data_independent_bound=False if median_normalize else True,
                    signature_is_truncated_approximation=kernel == 'signature_truncated_rbf',
                    signature_1d_endpoint_degeneracy=bool(kernel == 'signature_truncated_rbf'
                                                        and not time_aug and x.shape[2] == 1))
    return cost, metadata


def _logsumexp(values, axis):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.sum(np.exp(values-maximum), axis=axis, keepdims=True)), axis=axis)


def _marginal(value, size, name):
    result = np.full(size, 1/size) if value is None else np.asarray(value, dtype=np.float64)
    if (result.shape != (size,) or not np.isfinite(result).all() or np.any(result < 0)
            or not np.isclose(result.sum(), 1., atol=1e-12, rtol=1e-12)):
        raise ValueError(f'{name} must be a nonnegative probability vector of length {size}')
    return result / result.sum()


def sinkhorn_log(cost, epsilon=.05, max_iter=2000, tol=1e-8, a=None, b=None):
    """Solve entropic OT in float64 log space; fail explicitly on bad marginals.

    Zero-mass rows/columns are removed during solving and restored exactly.
    marginal_error is the maximum absolute row/column marginal discrepancy.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or min(cost.shape) < 1 or not np.isfinite(cost).all() or np.any(cost < 0):
        raise ValueError('cost must be a finite nonnegative nonempty matrix')
    if not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError('epsilon and tol must be positive finite values')
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError('max_iter must be a positive integer')
    a, b = _marginal(a, cost.shape[0], 'a'), _marginal(b, cost.shape[1], 'b')
    rows, cols = a > 0, b > 0
    active = cost[np.ix_(rows, cols)]
    # Row/column additive shifts leave the optimal coupling invariant.
    active = active - active.min(axis=1, keepdims=True)
    active = active - active.min(axis=0, keepdims=True)
    with np.errstate(over='ignore'):
        log_kernel = -active/epsilon
    if not np.isfinite(log_kernel).all():
        raise ValueError('cost/epsilon exceeds float64 range')
    log_a, log_b = np.log(a[rows]), np.log(b[cols])
    u, v = np.zeros_like(log_a), np.zeros_like(log_b)
    error = math.inf
    for iteration in range(1, max_iter+1):
        u = log_a-_logsumexp(log_kernel+v[None, :], axis=1)
        v = log_b-_logsumexp(log_kernel+u[:, None], axis=0)
        log_plan = log_kernel+u[:, None]+v[None, :]
        plan_active = np.exp(log_plan)
        error = float(max(np.max(np.abs(plan_active.sum(1)-a[rows])),
                          np.max(np.abs(plan_active.sum(0)-b[cols]))))
        if error <= tol:
            break
    diagnostics = dict(converged=bool(error <= tol), iterations=int(iteration),
                       marginal_error=error, epsilon=float(epsilon),
                       independent_cost=float(a @ cost @ b))
    if error > tol or not np.isfinite(error):
        raise SinkhornConvergenceError(diagnostics)
    plan = np.zeros_like(cost)
    plan[np.ix_(rows, cols)] = plan_active
    diagnostics['transport_cost'] = float(np.sum(plan*cost))
    return dict(plan=plan, **diagnostics)


def draw_plan_pairs(plan, count, seed=None, rng=None):
    """Draw IID pairs from the joint plan, never row-wise argmax."""
    plan = np.asarray(plan, dtype=np.float64)
    if (plan.ndim != 2 or min(plan.shape) < 1 or not np.isfinite(plan).all()
            or np.any(plan < 0) or not np.isfinite(plan.sum()) or plan.sum() <= 0):
        raise ValueError('plan must be a finite nonnegative matrix of positive mass')
    if not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError('count must be a nonnegative integer')
    if seed is not None and rng is not None:
        raise ValueError('provide either seed or rng, not both')
    rng = np.random.default_rng(seed) if rng is None else rng
    index = rng.choice(plan.size, size=count, p=plan.ravel()/plan.sum())
    return np.unravel_index(index, plan.shape)


def couple_endpoints(source, target, *, kernel='rbf', epsilon=.05, bandwidth=None,
                     derivative_weight=.01, signature_depth=3, time_aug=True,
                     median_normalize=True, max_iter=2000, tol=1e-8, count=None,
                     seed=None, rng=None):
    """Build a full empirical endpoint coupling and draw its endpoint indices."""
    cost, metadata = kernel_cost(source, target, kernel=kernel, bandwidth=bandwidth,
                                derivative_weight=derivative_weight, signature_depth=signature_depth,
                                time_aug=time_aug, median_normalize=median_normalize)
    solved = sinkhorn_log(cost, epsilon=epsilon, max_iter=max_iter, tol=tol)
    i, j = draw_plan_pairs(solved['plan'], len(source) if count is None else count, seed=seed, rng=rng)
    diagnostics = dict(metadata, **{key: value for key, value in solved.items() if key != 'plan'})
    diagnostics['conditional_source_independence_guaranteed'] = False
    return dict(source_indices=i, target_indices=j, plan=solved['plan'], cost=cost,
                diagnostics=diagnostics)
