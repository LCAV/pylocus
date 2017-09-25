#!/usr/bin/env python
# module TEST_ACD

import unittest
import numpy as np
from test_common import BaseCommon

from pylocus.point_set import PointSet
from pylocus.algorithms import reconstruct_acd

class TestACD(BaseCommon.TestAlgorithms):
    def setUp(self):
        self.create_points()

    def create_points(self, N=5, d=2):
        print('TestACD:create_points')
        self.pts = PointSet(N, d)
        self.pts.set_points('random')
        self.pts.init()
        self.index = 0

    def call_method(self):
        print('TestACD:call_method')
        Xhat, res = reconstruct_acd(self.pts.edm,
                                    real_points=self.pts.points,
                                    W=np.ones(self.pts.edm.shape),
                                    X0=self.pts.points,
                                    print_out=False, n_it=3)
        return Xhat


if __name__ == "__main__":
    unittest.main()
