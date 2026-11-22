"""Native-grid, single reference blade coefficients; no rotor Cp/Cq conversion.

Section arrays are [height, angle]; curves are [angle, (Cl,Cd,CM)].
CM is force-induced torque about the shaft, excluding section pitching moment.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

FEATURES = ('chord_m', 'thickness_ratio')
CONDITIONS = ('wind_ref_m_s', 'tsr_ref')
INPUT_CONTRACT = 'parameter-only-blade-v1'
LABEL_CONDITIONS = ('tsr_ref', 'wind_ref_m_s', 'chord_m', 'thickness_ratio')
TARGETS = ('Cl_blade', 'Cd_blade', 'CM_blade')
TARGET_CONTRACT = 'single-blade-local-lift-drag-force-induced-shaft-torque-v1'


def _finite(a, label):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f'{label}: non-finite values')
    return a


def parameter_inputs(wind_ref, tsr, chord_profile, thickness_profile, config):
    """Build physical parameter inputs without reading any solver output.

    Profiles are scalars or twenty spanwise values, constant over azimuth.
    Height centers and weights use the fixed uniform LF output grid.
    """
    from .resolution import midpoint_theta
    if config.get('nz',20) != 20:
        raise ValueError('single-blade parameter contract requires 20 heights')
    theta = midpoint_theta(config.get('ntheta',160))
    a = _finite([wind_ref,tsr], 'conditions')
    if a.shape != (2,) or np.any(a <= 0):
        raise ValueError('wind and TSR must be positive scalars')
    profiles = []
    for value,label in ((chord_profile,'chord'),(thickness_profile,'thickness ratio')):
        profile = _finite(value,label)
        if profile.ndim == 0:
            profile = np.full(20,float(profile))
        if profile.shape != (20,):
            raise ValueError(f'{label} must be a scalar or 20-layer vector')
        if np.any(profile <= 0) or (label == 'thickness ratio' and np.any(profile >= 1)):
            raise ValueError(f'{label} outside physical range')
        profiles.append(profile)
    height = float(config['height'])
    bottom = float(config.get('rotor_bottom_height',0.))
    if not np.isfinite([height,bottom]).all() or height <= 0:
        raise ValueError('height must be finite and positive')
    return dict(x=np.repeat(np.stack(profiles,axis=1)[:,:,None],len(theta),axis=2),
                a=a, z=bottom+(np.arange(20)+.5)*height/20,
                weights=np.full(20,height/20), theta=theta)


def integrate_blade(lift_per_length, drag_per_length, torque_per_length,
                    weights, wind_ref, chord, config):
    """Integrate [z,theta] force/torque densities using physical panel lengths."""
    arrays = [_finite(a, 'section loads') for a in
              (lift_per_length, drag_per_length, torque_per_length)]
    w = _finite(weights, 'weights')
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('section loads must have the same [height,angle] shape')
    if w.shape != (arrays[0].shape[0],) or np.any(w <= 0):
        raise ValueError('invalid physical height weights')
    rho, radius, height = (float(config[k]) for k in ('rho','radius','height'))
    if not np.isfinite([rho,radius,height,wind_ref,chord]).all() or min(rho,radius,height,wind_ref,chord) <= 0:
        raise ValueError('reference quantities must be finite and positive')
    if not np.isclose(w.sum(), height, atol=1e-5, rtol=1e-6):
        raise ValueError('height weights must span blade height')
    q = .5*rho*wind_ref**2
    denominators = (q*chord*height, q*chord*height, q*(2*radius*height)*radius)
    return np.stack([w@a/d for a,d in zip(arrays,denominators)], axis=-1)


def cosine_grid(config):
    n, h = int(config.get('nz',20)), float(config['height'])
    edges = h/2*(1-np.cos(np.arange(n+1)*np.pi/n))
    return (edges[:-1]+edges[1:])/2, np.diff(edges)


def validate_native_grid(theta_deg, z, config, *, cosine=True):
    nz, nt = int(config.get('nz',20)), int(config.get('ntheta',160))
    if (nz,nt) != (20,160):
        raise ValueError('single-blade contract requires 20 heights and 160 angles')
    expected = (np.arange(nt)+.5)*360/nt
    theta = _finite(theta_deg, 'angle')
    if theta.shape[-1] != nt or not np.allclose(theta, expected, atol=1e-5, rtol=0):
        raise ValueError('expected 160 unique native midpoint angles without phase interpolation')
    z = _finite(z, 'height')
    if z.ndim == 1:
        z = z[:,None]
    centers = cosine_grid(config)[0] if cosine else (
        float(config['rotor_bottom_height'])+(np.arange(nz)+.5)*float(config['height'])/nz)
    if z.shape[0] != nz or not np.allclose(z, centers[:,None], atol=6e-4 if cosine else 1e-6, rtol=0):
        raise ValueError('native height centers differ from expected panel geometry')


class _Inputs:
    """Read each input once; hash exactly the bytes subsequently parsed."""
    def __init__(self, root):
        self.root, self.hashes = Path(root), {}

    def text(self, filename):
        data = (self.root/filename).read_bytes()
        self.hashes[str(filename)] = hashlib.sha256(data).hexdigest()
        return data.decode('utf-8-sig')

    def json(self, filename):
        return json.loads(self.text(filename))

    def numeric(self, filename, fields):
        text = self.text(filename)
        header = text.splitlines()[0].split(',')
        ids = np.loadtxt(io.StringIO(text), delimiter=',', skiprows=1, usecols=(0,), dtype=str, ndmin=1)
        if not np.all(ids == self.root.name):
            raise ValueError('mixed case identities in '+str(filename))
        values = np.loadtxt(io.StringIO(text), delimiter=',', skiprows=1,
                            usecols=[header.index(f) for f in fields], ndmin=2)
        return _finite(values, str(filename))


def _config(config):
    for k,v in (('radius',55.),('height',112.),('rho',1.225)):
        if not np.isclose(float(config[k]),v,rtol=0,atol=1e-8):
            raise ValueError(f'fixed reference {k} mismatch')


def _conditions(info, case_id, config):
    if info['case_id'] != case_id:
        raise ValueError('case identity mismatch')
    a = _finite([info[k] for k in LABEL_CONDITIONS], 'label reference conditions')
    if np.any(a <= 0):
        raise ValueError('conditions must be positive')
    if not np.isclose(float(info['omega_rad_s']), a[0]*a[1]/float(config['radius']), rtol=1e-6):
        raise ValueError('omega/conditions reference mismatch')
    return a


def _result(inputs, info, a, theta, z, weights, y, **extra):
    public_a = a[[1,0]].copy()
    return dict(case_id=inputs.root.name, info=info, a=public_a, conditions=public_a.copy(),
                reference_conditions=a.copy(),
                theta=np.deg2rad(theta), z=z, weights=weights, y=y,
                metadata=dict(input_contract=INPUT_CONTRACT, target_contract=TARGET_CONTRACT,
                              reference_conditions_role='label normalization only; not model inputs',
                              phase='native blade1 midpoint angles; no shift or interpolation',
                              source_hashes=inputs.hashes), **extra)


def read_lf_case(case_path, config):
    _config(config)
    inputs = _Inputs(case_path)
    state = inputs.json('state.json')
    if state.get('status') != 'completed' or state.get('converged') is not True:
        raise ValueError('LF case must be completed and converged')
    import csv
    infos = list(csv.DictReader(io.StringIO(inputs.text('case_info.csv'))))
    if len(infos) != 1:
        raise ValueError('case_info must contain one identity')
    info = infos[0]
    a = _conditions(info, inputs.root.name, config)
    parameters = parameter_inputs(a[1],a[0],a[2],a[3],config)
    fields = ('layer_id','azimuth_id','z_ground_m','theta_deg')+(
        'lift_N_per_m','drag_N_per_m','torque_Nm_per_m')
    rows = inputs.numeric('section_performance.csv', fields)
    if rows.shape[0] != 20*160:
        raise ValueError('incomplete LF section grid')
    rows = rows[np.lexsort((rows[:,1], rows[:,0]))]
    expected = np.array([(z,t) for z in range(1,21) for t in range(1,161)])
    if not np.array_equal(rows[:,:2],expected):
        raise ValueError('duplicate or invalid LF grid ids')
    grid = rows.reshape(20,160,-1)
    validate_native_grid(grid[:,:,3],grid[:,:,2],config,cosine=False)
    weights = np.full(20,float(config['height'])/20)
    y = integrate_blade(grid[:,:,-3],grid[:,:,-2],grid[:,:,-1],weights,a[1],a[2],config)
    return _result(inputs, info, a, grid[0,:,3], grid[:,0,2], weights, y,
                   x=parameters['x'], baseline=y.copy())


def read_hf_case(case_path, config, blade_id=1):
    _config(config)
    if blade_id != 1:
        raise ValueError('only native reference blade1 is authorized')
    inputs = _Inputs(case_path)
    model = inputs.json('model_metadata.json')
    run = inputs.json('run_metadata.json')
    if run.get('completed') is not True:
        raise ValueError('HF case must be completed')
    info = model['case']
    a = _conditions(info, inputs.root.name, config)
    polar = model.get('polar_metadata')
    if polar is not None:
        for key, expected in (('rotor_radius_m',config['radius']),('rotor_height_m',config['height'])):
            if not np.isclose(float(polar[key]),float(expected)):
                raise ValueError('HF geometry reference mismatch: '+key)
    # Original NACA0018 polar cases have no polar_metadata. Check the actual
    # native simulation definition for every case, including density and U_ref.
    simulation = inputs.text('origin.sim')
    native = {}
    for line in simulation.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in ('DENSITYAIR','MEANINF','RPMPRESCRIBED'):
            native[parts[1]] = float(parts[0])
    for key, expected in (('DENSITYAIR',config['rho']),('MEANINF',a[1]),
                          ('RPMPRESCRIBED',float(info['omega_rad_s'])*60/(2*np.pi))):
        if key not in native or not np.isclose(native[key],float(expected)):
            raise ValueError('HF condition/reference mismatch: '+key)
    geometry = inputs.text('New_Blade_Turb/Aero/New_Blade.bld')
    nodes = []
    for line in geometry.splitlines():
        parts = line.split()
        try:
            node = [float(v) for v in parts[:3]]
        except ValueError:
            continue
        if len(parts) >= 8 and len(node) == 3:
            nodes.append(node)
    nodes = np.asarray(nodes)
    if nodes.shape[0] < 2 or not np.allclose(nodes[:,1],a[2]) or not np.allclose(nodes[:,2],config['radius']) or not np.isclose(np.ptp(nodes[:,0]),config['height']):
        raise ValueError('HF actual blade chord/radius/height mismatch')
    fields = ('reference_azimuth_deg','blade_id','panel_id','blade_azimuth_deg',
              'Height [m]','Lift Coefficient [-]','Drag Coefficient [-]','Total Velocity [m/s]')
    section = inputs.numeric('section_azimuth.csv',fields)
    section = section[section[:,1] == blade_id]
    if section.shape[0] != 20*160:
        raise ValueError('incomplete HF section grid')
    section = section[np.lexsort((section[:,3],section[:,2]))]
    expected = np.repeat(np.arange(20),160)
    if not np.array_equal(section[:,2],expected):
        raise ValueError('invalid HF panel ids')
    grid = section.reshape(20,160,-1)
    validate_native_grid(grid[:,:,3],grid[:,:,4],config)
    if not np.allclose(grid[:,:,0],grid[:,:,3],atol=1e-5,rtol=0):
        raise ValueError('reference and native blade1 angle mismatch')
    blade = inputs.numeric('blade_azimuth.csv',('reference_azimuth_deg','blade_id',
                           'Azimuthal Angle [deg]','Tangential Load [N]'))
    blade = blade[blade[:,1] == blade_id]
    blade = blade[np.argsort(blade[:,2])]
    if blade.shape != (160,4):
        raise ValueError('incomplete native blade load grid')
    validate_native_grid(blade[:,2],grid[:,:,4],config)
    if not np.allclose(blade[:,0],blade[:,2],atol=1e-5,rtol=0):
        raise ValueError('native blade phase mismatch')
    _, weights = cosine_grid(config)
    local_q = .5*float(config['rho'])*grid[:,:,7]**2
    y = integrate_blade(local_q*grid[:,:,5]*a[2],local_q*grid[:,:,6]*a[2],
                        np.zeros((20,160)),weights,a[1],a[2],config)
    q_ref = .5*float(config['rho'])*a[1]**2
    y[:,2] = blade[:,3]*float(config['radius'])/(q_ref*(2*float(config['radius'])*float(config['height']))*float(config['radius']))
    return _result(inputs,info,a,blade[:,2],grid[:,0,4],weights,y)


def match_cases(lf, hf):
    """Reject mismatched identities, physical conditions, and native angle grids."""
    if lf['case_id'] != hf['case_id']:
        raise ValueError('LF/HF case identity mismatch')
    if not np.allclose(lf['conditions'],hf['conditions'],rtol=1e-6,atol=1e-7):
        raise ValueError('LF/HF physical condition mismatch')
    def geometry(case):
        if 'info' in case:
            return [float(case['info'][key]) for key in FEATURES]
        if 'reference_conditions' in case:
            return np.asarray(case['reference_conditions'])[2:4]
        if np.asarray(case['conditions']).shape == (4,):
            return np.asarray(case['conditions'])[2:4]
        raise ValueError('LF/HF geometry reference unavailable')
    if not np.allclose(geometry(lf),geometry(hf),rtol=1e-6,atol=1e-7):
        raise ValueError('LF/HF physical geometry condition mismatch')
    # A float32 cache rounds native radians by up to about 2.4e-7.
    # This tolerance accommodates storage precision without changing phase.
    if not np.allclose(lf['theta'],hf['theta'],rtol=0,atol=1e-6):
        raise ValueError('LF/HF native angle mismatch')
    return True
