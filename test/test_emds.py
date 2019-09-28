#!/usr/bin/env python
# module TEST_EMDS

import unittest
import numpy as np
from .test_common import BaseCommon

from pylocus.point_set import HeterogenousSet
from pylocus.algorithms import reconstruct_emds


class TestEMDS(BaseCommon.TestAlgorithms):
    def setUp(self):
        print('TestEMDS:setUp')
        BaseCommon.TestAlgorithms.setUp(self)
        self.n_it = 5
        self.N_zero = [5]
        self.N_relaxed = [5]
        self.eps = 1e-7
        self.methods = ['relaxed','iterative']

    def create_points(self, N=5, d=3):
        print('TestEMDS:create_points')
        self.pts = HeterogenousSet(N, d)
        self.pts.set_points('normal')
        self.C, self.b = self.pts.get_KE_constraints()

    def call_method(self, method=''):
        print('TestEMDS:call_method with', method)
        if method == '':
            return reconstruct_emds(self.pts.edm, Om=self.pts.Om,
                                    all_points=self.pts.points)
        else:
            return reconstruct_emds(self.pts.edm, Om=self.pts.Om,
                                    all_points=self.pts.points, C=self.C, b=self.b, method=method)


if __name__ == "__main__":
    unittest.main()
