#!/usr/bin/env python
# module TEST_COMPLETION

import unittest
import numpy as np

from pylocus.point_set import PointSet
from pylocus.edm_completion import *
    
class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.pts = PointSet(N=5, d=3)
        self.pts.set_points(mode='random')

    def test_completion(self):
        edm_complete = self.pts.edm.copy()
        W = np.ones(edm_complete.shape)
        indices_missing = [[4,1,2],[0,3,1]]
        W[indices_missing] = 0.0
        W = np.multiply(W, W.T)
        print(W)
        X0 = self.pts.points
        X0 += np.random.normal(loc=0.0, scale=0.1)
        edm = completion_acd(edm_complete, X0)
        np.testing.assert_allclose(edm_complete, edm)
        
        edm_missing = np.multiply(edm_complete, W)
        edm = completion_acd(edm_missing, X0, W)
        np.testing.assert_allclose(edm_complete, edm)

if __name__ == "__main__":
    unittest.main()
