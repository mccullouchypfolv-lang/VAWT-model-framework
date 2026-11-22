import copy
import json
import unittest
import csv
import io
from unittest.mock import patch
from pathlib import Path

import numpy as np

from lf_operator import blade_data as bd


ROOT = Path(__file__).resolve().parents[1]
LF = ROOT.parent/'仿真/dataset/outputs/results_PC_momentum_c1'
HF = ROOT.parent/'仿真/High-fidelity/num49/results_origin_detailed/results_origin_detailed'


class BladeDataTests(unittest.TestCase):
    def test_matching_accepts_cached_float32_angles_only_within_roundoff(self):
        theta = (np.arange(160)+.5)*2*np.pi/160
        hf = dict(case_id='case', conditions=np.array([9.,2.]),
                  reference_conditions=np.array([2.,9.,2.,.18]),theta=theta)
        lf = dict(hf, theta=theta.astype(np.float32))
        self.assertTrue(bd.match_cases(lf,hf))
        with self.assertRaisesRegex(ValueError,'angle'):
            bd.match_cases(dict(lf,theta=lf['theta']+1e-4),hf)

    def test_parameter_inputs_independent_variable_profiles(self):
        self.assertTrue(hasattr(bd, 'parameter_inputs'))
        config = dict(height=112., rotor_bottom_height=3., nz=20, ntheta=160)
        chord = np.linspace(1., 3., 20)
        thickness = np.linspace(.1, .3, 20)
        result = bd.parameter_inputs(9., 2.5, chord, thickness, config)
        self.assertEqual(result['x'].shape, (20,2,160))
        np.testing.assert_array_equal(result['a'], [9.,2.5])
        np.testing.assert_array_equal(result['x'][:,:,0], np.column_stack([chord,thickness]))
        np.testing.assert_array_equal(result['x'], np.repeat(result['x'][:,:,:1],160,axis=2))
        np.testing.assert_allclose(result['z'], 3.+(np.arange(20)+.5)*5.6)
        np.testing.assert_allclose(result['weights'], np.full(20,5.6))
        np.testing.assert_allclose(result['theta'], (np.arange(160)+.5)*2*np.pi/160)
        scalar = bd.parameter_inputs(9.,2.5,2.,.18,config)
        np.testing.assert_array_equal(scalar['x'][:,0], np.full((20,160),2.))

    def test_parameter_inputs_reject_nonphysical_or_angle_varying_values(self):
        self.assertTrue(hasattr(bd, 'parameter_inputs'))
        config = dict(height=112., rotor_bottom_height=3.)
        for wind,tsr,chord,thickness in [(0,2,2,.18),(9,-2,2,.18),
                (9,2,[2]*19,.18),(9,2,np.ones((20,160)),.18),
                (9,2,0,.18),(9,2,2,1.),(9,2,2,0.),(9,2,2,np.nan)]:
            with self.subTest(values=(wind,tsr)), self.assertRaises(ValueError):
                bd.parameter_inputs(wind,tsr,chord,thickness,config)

    @unittest.skipUnless((LF/'common_config.json').exists(), 'raw data unavailable')
    def test_lf_inputs_ignore_all_solver_feature_columns(self):
        config = json.loads((LF/'common_config.json').read_text(encoding='utf-8-sig'))
        source = next(p for p in (LF/'cases').iterdir() if p.is_dir() and (p/'section_performance.csv').exists())
        original = bd.read_lf_case(source,config)
        with (source/'section_performance.csv').open(encoding='utf-8-sig',newline='') as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames,list(reader)
        permitted = {'case_id','layer_id','azimuth_id','z_ground_m','theta_deg',
                     'lift_N_per_m','drag_N_per_m','torque_Nm_per_m'}
        for row in rows:
            for key in fields:
                if key not in permitted:
                    row[key] = 'nan'
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        original_text = bd._Inputs.text
        def perturbed_text(inputs, filename):
            text = original_text(inputs, filename)
            return buffer.getvalue() if filename == 'section_performance.csv' else text
        with patch.object(bd._Inputs,'text',perturbed_text):
            changed = bd.read_lf_case(source,config)
        self.assertEqual(changed['x'].shape,(20,2,160))
        for key in ('x','a','y','theta','z','weights'):
            np.testing.assert_array_equal(changed[key],original[key])
        self.assertEqual(changed['metadata']['input_contract'],'parameter-only-blade-v1')

    def test_unequal_weights_and_reference_denominators(self):
        config = dict(rho=2., radius=5., height=4.)
        lift = np.array([[2., -2.], [8., -8.]])
        drag = np.ones((2, 2))*3
        torque = np.array([[10., -20.], [40., -80.]])
        actual = bd.integrate_blade(lift, drag, torque, [1., 3.], 2., 2., config)
        expected = np.array([[26/32, 12/32, 130/800], [-26/32, 12/32, -260/800]])
        np.testing.assert_allclose(actual, expected)
        self.assertLess(actual[1, 0], 0)
        self.assertFalse(np.allclose(actual[:, 0], 5*actual[:, 2]))

    def test_native_grid_validation(self):
        config = dict(nz=20, ntheta=160, height=112.)
        edges = 56*(1-np.cos(np.arange(21)*np.pi/20))
        z, weights = bd.cosine_grid(config)
        np.testing.assert_allclose(z, (edges[:-1]+edges[1:])/2)
        np.testing.assert_allclose(weights, np.diff(edges))
        theta = (np.arange(160)+.5)*2.25
        bd.validate_native_grid(theta, np.tile(z[:, None], (1,160)), config)
        with self.assertRaisesRegex(ValueError, 'angle'):
            bd.validate_native_grid(theta+1., z[:,None], config)
        with self.assertRaisesRegex(ValueError, 'height'):
            bd.validate_native_grid(theta, (z+1)[:,None], config)

    @unittest.skipUnless((LF/'common_config.json').exists() and HF.exists(), 'raw data unavailable')
    def test_actual_first_case_and_matching(self):
        config = json.loads((LF/'common_config.json').read_text(encoding='utf-8-sig'))
        case_id = sorted(p.name for p in HF.iterdir() if p.is_dir())[0]
        lf = bd.read_lf_case(LF/'cases'/case_id, config)
        hf = bd.read_hf_case(HF/case_id, config)
        self.assertEqual(lf['x'].shape, (20,2,160))
        self.assertEqual(lf['a'].shape,(2,))
        self.assertEqual(hf['y'].shape, (160,3))
        self.assertEqual(tuple(bd.TARGETS), ('Cl_blade','Cd_blade','CM_blade'))
        np.testing.assert_array_equal(lf['baseline'], lf['y'])
        self.assertAlmostEqual(float(hf['weights'].sum()), 112., places=5)
        bd.match_cases(lf,hf)
        self.assertTrue(all(len(v)==64 for v in hf['metadata']['source_hashes'].values()))
        broken = copy.deepcopy(hf)
        broken['conditions'][0] += .25
        with self.assertRaisesRegex(ValueError, 'condition'):
            bd.match_cases(lf, broken)
        broken = dict(hf, case_id='wrong')
        with self.assertRaisesRegex(ValueError, 'identity'):
            bd.match_cases(lf, broken)

    @unittest.skipUnless((HF/'S000088_L015').exists(), 'raw data unavailable')
    def test_original_polar_case_has_no_polar_metadata(self):
        config = json.loads((LF/'common_config.json').read_text(encoding='utf-8-sig'))
        hf = bd.read_hf_case(HF/'S000088_L015',config)
        self.assertEqual(hf['y'].shape,(160,3))
        self.assertTrue(np.isfinite(hf['y']).all())

    @unittest.skipUnless(HF.exists(), 'raw data unavailable')
    def test_hf_native_force_reference_independent_reconstruction(self):
        import csv
        config = json.loads((LF/'common_config.json').read_text(encoding='utf-8-sig'))
        case = sorted(p for p in HF.iterdir() if p.is_dir())[0]
        item = bd.read_hf_case(case,config)
        with (case/'section_azimuth.csv').open() as f:
            sections = [r for r in csv.DictReader(f) if r['blade_id']=='1' and r['blade_azimuth_deg']=='1.125']
        qref = .5*config['rho']*item['conditions'][0]**2
        chord = item['reference_conditions'][2]
        expected_cl = sum(.5*config['rho']*float(r['Total Velocity [m/s]'])**2*float(r['Lift Coefficient [-]'])*chord*item['weights'][int(r['panel_id'])] for r in sections)/(qref*chord*112)
        with (case/'blade_azimuth.csv').open() as f:
            blade = next(r for r in csv.DictReader(f) if r['blade_id']=='1' and r['reference_azimuth_deg']=='1.125')
        expected_cm = float(blade['Tangential Load [N]'])*55/(qref*(2*55*112)*55)
        np.testing.assert_allclose(item['y'][0,[0,2]],[expected_cl,expected_cm])


if __name__ == '__main__':
    unittest.main()
