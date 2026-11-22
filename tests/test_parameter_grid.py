"""Tensor-product evaluation over physical conditions and angular coordinates."""
import unittest
import numpy as np


class ParameterGridTests(unittest.TestCase):
    def test_three_axes_shape_and_stable_condition_order(self):
        from lf_operator.parameter_grid import build_parameter_grid
        spec=dict(wind_ref_m_s={'min':8.,'max':20.,'count':8},tsr_ref={'min':1.5,'max':6.,'count':8},
                  chord_m=4.,thickness_ratio=.18,n_theta=8)
        grid=build_parameter_grid(spec,n_theta=64)
        self.assertEqual(grid['shape'],(8,8,64,3))
        self.assertEqual(len(grid['queries']),64)
        self.assertEqual(grid['queries'][0]['wind_ref_m_s'],8.)
        self.assertEqual(grid['queries'][7]['tsr_ref'],6.)
        self.assertEqual(grid['queries'][8]['tsr_ref'],1.5)
        spec['wind_ref_m_s']['count']=64;spec['tsr_ref']['count']=64
        grid=build_parameter_grid(spec,n_theta=64)
        self.assertEqual(grid['shape'],(64,64,64,3))
        self.assertEqual(len(grid['queries']),4096)

    def test_axes_and_geometry_contract_validation(self):
        from lf_operator.parameter_grid import build_parameter_grid
        spec=dict(wind_ref_m_s=[8.,10.],tsr_ref=[2.,3.],chord_m=[4.]*20,thickness_ratio=.18,n_theta=64)
        grid=build_parameter_grid(spec)
        np.testing.assert_array_equal(grid['wind_ref_m_s'],[8.,10.])
        for patch in [dict(wind_ref_m_s=[0.,10.]),dict(tsr_ref=[3.,2.]),dict(chord_m=[4.]*19),
                      dict(thickness_ratio=1.2),dict(n_theta=1),dict(typo=1)]:
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                build_parameter_grid(dict(spec,**patch))


if __name__=='__main__':unittest.main()
