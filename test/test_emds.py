#!/usr/bin/env python
# module TEST_EMDS

import unittest
import numpy as np
from pylocus.point_set import HeterogenousSet
from pylocus.algorithms import reconstruct_emds

class TestEMDS(unittest.TestCase):
    def setUp(self, N=5, d=2):
        self.pts = HeterogenousSet(N, d)
        self.pts.set_points('normal')

    def test_multiple(self):
        for i in range(100):
            self.test_zero_noise()

    def test_zero_noise(self):
        for N in range(4, 10):
            for d in (2,3):
                self.setUp(N, d)
                Y = reconstruct_emds(self.pts.edm, Om=self.pts.Om, 
                                     real_points=self.pts.points)
                self.assertTrue(np.linalg.norm(self.pts.points-Y) < 1e-13, 
                                'noiseless case did not give 0 error.')

if __name__ == "__main__":
    unittest.main()
