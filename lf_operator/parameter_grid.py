"""Chunked tensor-product evaluation of the existing conditional operators."""
from pathlib import Path

import numpy as np

from .blade_data import INPUT_CONTRACT, TARGET_CONTRACT, parameter_inputs
from .resolution import midpoint_theta


def _axis(value, name):
    if isinstance(value, dict):
        if set(value) != {'min', 'max', 'count'}:
            raise ValueError(f'{name} range requires only min, max and count')
        count = value['count']
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError(f'{name} count must be a positive integer')
        try:
            bounds = np.asarray([value['min'], value['max']], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} bounds must be numeric') from exc
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or np.any(bounds <= 0) or bounds[0] > bounds[1]:
            raise ValueError(f'{name} requires positive finite min <= max')
        values = np.linspace(*bounds, count)
    else:
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(f'{name} must be an axis array or min/max/count range')
        try:
            values = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be numeric') from exc
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.any(values <= 0) or np.any(np.diff(values) <= 0):
        raise ValueError(f'{name} axis must be positive, finite and strictly increasing')
    return values


def build_parameter_grid(spec, n_theta=None):
    """Expand U-major/TSR-minor queries with one shared blade geometry."""
    required = {'wind_ref_m_s', 'tsr_ref', 'chord_m', 'thickness_ratio'}
    if not isinstance(spec, dict) or not required <= spec.keys() or set(spec) - required - {'n_theta'}:
        raise ValueError('grid requires wind_ref_m_s, tsr_ref, chord_m, thickness_ratio and optional n_theta only')
    wind = _axis(spec['wind_ref_m_s'], 'wind_ref_m_s')
    tsr = _axis(spec['tsr_ref'], 'tsr_ref')
    theta = midpoint_theta(n_theta if n_theta is not None else spec.get('n_theta', 64))
    # Validate profiles against the same physical input contract as inference.
    validated = parameter_inputs(wind[0], tsr[0], spec['chord_m'], spec['thickness_ratio'],
                                 dict(nz=20, ntheta=len(theta), height=112., rotor_bottom_height=59.6))
    chord = validated['x'][:, 0, 0].tolist()
    thickness = validated['x'][:, 1, 0].tolist()
    queries = [dict(wind_ref_m_s=float(u), tsr_ref=float(t), chord_m=chord,
                    thickness_ratio=thickness) for u in wind for t in tsr]
    return dict(queries=queries, wind_ref_m_s=wind, tsr_ref=tsr, theta=theta,
                shape=(len(wind), len(tsr), len(theta), 3))


def predict_grid(cfg, spec, output, n_theta=None, lf_only=False, query_batch_size=8):
    """Evaluate conditions in bounded batches, retaining curve summaries only.

    Each chunk uses seed + its condition offset. Changing query_batch_size can
    change sampled uncertainty; the fitted source and all weights stay fixed.
    """
    import torch
    from . import blade_pipeline as pipeline
    from .conditioning import standardize_conditions, tensor_conditions
    from .data import digest
    from .model import create_lf_model as FiLMFNO
    from .residual_flow import create_velocity_model as ConditionalVelocity

    if (isinstance(query_batch_size, (bool, np.bool_))
            or not isinstance(query_batch_size, (int, np.integer)) or query_batch_size < 1):
        raise ValueError('query_batch_size must be a positive integer')
    out = Path(output)
    if out.exists() or out.suffix != '.npz':
        raise ValueError('output must be a new .npz path')
    grid = build_parameter_grid(spec, n_theta=n_theta)
    device = pipeline.setup(cfg)
    lf_path = pipeline.ROOT / cfg['work'] / 'stage1' / 'best.pt'
    lf_checkpoint = torch.load(lf_path, map_location='cpu', weights_only=False)
    pipeline.validate_parameter_contract(lf_checkpoint)
    lf_model = FiLMFNO(**lf_checkpoint['model_config']).to(device).eval()
    lf_model.load_state_dict(lf_checkpoint['model'])
    if not lf_only:
        checkpoint = torch.load(pipeline.flow_output_directory(cfg) / 'best.pt', map_location='cpu', weights_only=False)
        pipeline.validate_velocity_checkpoint(checkpoint)
        if (checkpoint.get('input_contract') != INPUT_CONTRACT
                or checkpoint.get('targets') != pipeline.TARGETS
                or checkpoint.get('target_contract') != TARGET_CONTRACT
                or checkpoint.get('lf_checkpoint_sha256') != digest(lf_path)):
            raise ValueError('input/target or LF checkpoint mismatch')
        model = ConditionalVelocity(**checkpoint['model_config']).to(device).eval()
        model.load_state_dict(checkpoint['model'])
    count = len(grid['queries'])
    rotor_only=not lf_only and checkpoint['model_config'].get('architecture')=='rotor-only-filmfno-v1'
    keys = ('prediction',) if lf_only else ('mean', 'std', 'q05', 'q95')
    summaries = {key: np.empty((count, len(grid['theta']), 3), dtype=np.float32) for key in keys}
    for offset in range(0, count, query_batch_size):
        end = min(offset + query_batch_size, count)
        arrays = pipeline.lf_predict_parameters(lf_checkpoint, grid['queries'][offset:end], device,
                                                n_theta=len(grid['theta']), model=lf_model,return_field_features=not rotor_only)
        if lf_only:
            summaries['prediction'][offset:end] = arrays['lf_prediction']
        else:
            tensors = tensor_conditions(standardize_conditions(arrays, checkpoint['scalers']), arrays, device,include_height=not rotor_only)
            result = pipeline.draw_predictions(model, tensors, arrays, np.arange(end-offset),
                checkpoint['source'], checkpoint['residual_stats'], cfg['evaluation_draws'],
                cfg['ode_steps'], cfg['seed'] + offset, reference_theta=np.asarray(checkpoint['theta']),**pipeline.ode_options(cfg))
            for key in keys:
                summaries[key][offset:end] = result[key]
            # Release draws and chunk features before constructing the next chunk.
            del result, tensors
        del arrays
    metadata = dict(wind_ref_m_s=grid['wind_ref_m_s'], tsr_ref=grid['tsr_ref'], theta=grid['theta'],
                    targets=np.asarray(pipeline.TARGETS), input_contract=np.asarray(INPUT_CONTRACT),
                    axis_order=np.asarray(['wind_ref_m_s', 'tsr_ref', 'theta', 'target']),
                    shape=np.asarray(grid['shape']))
    if not lf_only:
        from .resolution import source_scope
        metadata.update(training_theta=np.asarray(checkpoint['theta']), source_scope=np.asarray(source_scope(checkpoint['source'])))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{key: values.reshape(grid['shape']) for key, values in summaries.items()}, **metadata)
    return dict(output=str(out.resolve()), shape=list(grid['shape']), conditions=count, n_theta=len(grid['theta']))
