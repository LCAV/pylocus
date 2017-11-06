#!/usr/bin/env python
# module TEST_EMDS

import unittest
import numpy as np
from test_common import BaseCommon

from pylocus.point_set import PointSet
from pylocus.algorithms import reconstruct_dwmds, reconstruct_acd


class TestACD(BaseCommon.TestAlgorithms):
    def setUp(self):
        print('TestACD:setUp')
        BaseCommon.TestAlgorithms.setUp(self)
        self.n_it = 10
        self.N_zero = [5]
        self.N_relaxed = [5]
        self.eps = 1e-8

    def create_points(self, N=10, d=3):
        print('TestACD:create_points')
        self.pts = PointSet(N, d)
        self.pts.set_points('normal')
        self.X0 = self.pts.points + np.random.normal(1.0, size=self.pts.points.shape)

    def call_method(self, method=''):
        print('TestACD:call_method')
        tol = 1e-5
        Xhat, __, __ = reconstruct_acd(self.pts.edm, X0=self.X0, print_out=False, tol=tol)
        return Xhat

    def test_decreasing_cost(self, method=''):
        tol = 1e-5
        self.create_points()
        __, costs, __ = reconstruct_acd(self.pts.edm, X0=self.X0, print_out=False, tol=tol)
        cprev = np.inf
        for c in costs:
            self.assertTrue(c <= cprev+tol, 'c: {}, cprev:{}'.format(c, cprev))
            cprev = c

if __name__ == "__main__":
    unittest.main()
