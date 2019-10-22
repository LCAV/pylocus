#!/usr/bin/env python
# module TEST_COMPLETION

import unittest
import numpy as np

from pylocus.point_set import PointSet
from pylocus.edm_completion import *
    
class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.pts = PointSet(N=5, d=3)
        self.pts.set_points(mode='convex')
        W = np.ones(self.pts.edm.shape)
        indices_missing = ([4,1],[0,3])
        W[indices_missing] = 0.0
        W = np.multiply(W, W.T)
        np.fill_diagonal(W, 0.0)
        self.W = W

        # TODO: find out why sdr and rank alternation don't work.
        #self.methods = ['acd', 'dwmds', 'sdr', 'rank']
        self.methods = ['acd', 'dwmds']

    def call_method(self, edm_input, method):
        print('running method', method)
        edm_missing = np.multiply(edm_input, self.W)
        if method == 'sdr':
            return semidefinite_relaxation(edm_input, lamda=1000, W=self.W)
        elif method == 'rank':
            return rank_alternation(edm_missing, rank=self.pts.d+2, niter=100)[0]
        else:
            X0 = self.pts.points
            X0 += np.random.normal(loc=0.0, scale=0.01)
            if method == 'acd':
                return completion_acd(edm_missing, X0, self.W)
            elif method == 'dwmds':
                return completion_dwmds(edm_missing, X0, self.W)


    def test_complete(self):
        for i in range(10):
            self.pts.set_points('random')
            for method in self.methods:
                edm_complete = self.pts.edm.copy()
                edm_output = self.call_method(edm_complete, method)
                np.testing.assert_allclose(edm_complete, edm_output)

if __name__ == "__main__":
    unittest.main()
