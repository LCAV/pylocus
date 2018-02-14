#!/usr/bin/env python
# module TEST_SRLS

import unittest
import numpy as np
from test_common import BaseCommon

from pylocus.point_set import PointSet, create_from_points
from pylocus.simulation import create_noisy_edm
from pylocus.algorithms import reconstruct_srls


class TestSRLS(BaseCommon.TestAlgorithms):
    def setUp(self):
        BaseCommon.TestAlgorithms.setUp(self)
        self.create_points()

    def create_points(self, N=10, d=2):
        print('TestSRLS:create_points')
        self.pts = PointSet(N, d)
        self.pts.set_points('random')
        # example point set that fails:
        # self.pts.points = np.array([[0.63, 0.45],
        #  [0.35, 0.37],
        #  [0.69, 0.71],
        #  [0.71, 0.73],
        #  [0.43, 0.44],
        #  [0.58, 0.59]])
        self.pts.init()
        self.n = 1

    def call_method(self, method=''):
        print('TestSRLS:call_method')
        return reconstruct_srls(self.pts.edm, self.pts.points, n=self.n,
                                W=np.ones(self.pts.edm.shape))

    def test_multiple_weights(self):
        print('TestSRLS:test_multiple')
        for i in range(100):
            self.create_points()
            self.zero_weights(0.0)
            self.zero_weights(0.1)
            self.zero_weights(1.0)

    def zero_weights(self, noise=0.1):
        print('TestSRLS:test_zero_weights({})'.format(noise))
        index = np.arange(self.n)
        other = np.delete(range(self.pts.N), index)
        edm_noisy = create_noisy_edm(self.pts.edm, noise)

        # missing anchors
        N_missing = 2
        indices = np.random.choice(other, size=N_missing, replace=False)
        reduced_points = np.delete(self.pts.points, indices, axis=0)
        points_missing = create_from_points(reduced_points, PointSet)
        edm_anchors = np.delete(edm_noisy, indices, axis=0)
        edm_anchors = np.delete(edm_anchors, indices, axis=1)
        missing_anchors = reconstruct_srls(
            edm_anchors, points_missing.points, n=self.n, W=None,
            print_out=False)

        # missing distances
        weights = np.ones(edm_noisy.shape)
        weights[indices, index] = 0.0
        weights[index, indices] = 0.0

        missing_distances = reconstruct_srls(
            edm_noisy, self.pts.points, n=self.n, W=weights)
        left_distances = np.delete(range(self.pts.N), indices)

        self.assertTrue(np.linalg.norm(
            missing_distances[left_distances, :] - missing_anchors) < 1e-10, 'anchors moved.')
        self.assertTrue(np.linalg.norm(
            missing_distances[index, :] - missing_anchors[index, :]) < 1e-10, 'point moved.')
        if noise == 0.0:
            error = np.linalg.norm(missing_anchors - points_missing.points)
            u, s, v = np.linalg.svd(self.pts.points[other, :])
            try:
                self.assertTrue(error < 1e-10, 'error {}'.format(error))
            except:
                print('failed with anchors (this can be due to unlucky geometry):',
                      points_missing.points)
                raise


if __name__ == "__main__":
    unittest.main()
