#!/usr/bin/env python
# module TEST_SDP

import unittest
import numpy as np
from .test_common import BaseCommon

from pylocus.point_set import PointSet
from pylocus.algorithms import reconstruct_sdp


class TestSDP(BaseCommon.TestAlgorithms):
    def setUp(self):
        print('TestSDP:setUp')
        BaseCommon.TestAlgorithms.setUp(self)
        self.eps = 1e-8
        self.success_rate = 50
        self.n_it = 1
        self.N_zero = [6]
        self.N_relaxed = [6]

    def create_points(self, N=10, d=3):
        print('TestSDP:create_points')
        self.pts = PointSet(N, d)
        self.pts.set_points('normal')

    def call_method(self, method=''):
        print('TestSDP:call_method')
        Xhat, edm = reconstruct_sdp(self.pts.edm, all_points=self.pts.points,
                                    solver='CVXOPT', method='maximize')
        return Xhat

    def test_parameters(self):
        print('TestSDP:test_parameters')
        self.create_points()
        epsilons = [1e-3, 1e-5, 1e-7]
        options_list = [{},
                        {'solver': 'CVXOPT',
                         'abstol': 1e-5,
                         'reltol': 1e-6,
                         'feastol': 1e-7}]
        for options, eps in zip(options_list, epsilons):
            print('testing options', options, eps)
            self.eps = eps
            points_estimate, __ = reconstruct_sdp(
                self.pts.edm, all_points=self.pts.points, method='maximize', **options)
            error = np.linalg.norm(self.pts.points - points_estimate)
            self.assertTrue(error < self.eps, 'with options {} \nerror: {} not smaller than {}'.format(
                options, error, self.eps))


if __name__ == "__main__":
    unittest.main()
