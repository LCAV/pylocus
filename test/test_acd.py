#!/usr/bin/env python
# module TEST_ACD

import unittest
import numpy as np
from .test_common import BaseCommon

from pylocus.point_set import PointSet
from pylocus.algorithms import reconstruct_acd
from pylocus.simulation import create_noisy_edm

class TestACD(BaseCommon.TestAlgorithms):
    def setUp(self):
        BaseCommon.TestAlgorithms.setUp(self)
        self.create_points()
        self.n_it = 10

    def create_points(self, N=5, d=2):
        print('TestACD:create_points')
        self.pts = PointSet(N, d)
        self.pts.set_points('random')
        self.pts.init()
        self.index = 0

    def call_method(self, method=''):
        print('TestACD:call_method')
        Xhat, costs = reconstruct_acd(self.pts.edm,
                                    W=np.ones(self.pts.edm.shape),
                                    X0=self.pts.points,
                                    print_out=False, sweeps=3)
        return Xhat

    def add_noise(self, noise=1e-6):
        self.pts.edm = create_noisy_edm(self.pts.edm, noise)


if __name__ == "__main__":
    unittest.main()
