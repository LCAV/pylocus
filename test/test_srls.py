#!/usr/bin/env python
# module TEST_SRLS

import unittest
import numpy as np
from test_common import BaseCommon

from pylocus.point_set import PointSet, create_from_points
from pylocus.simulation import create_noisy_edm
from pylocus.algorithms import reconstruct_srls
from pylocus.lateration import SRLS

class TestSRLS(BaseCommon.TestAlgorithms):
    #class TestSRLS(unittest.TestCase):
    def setUp(self):
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
        self.index = 0

    def call_method(self):
        print('TestSRLS:call_method')
        return reconstruct_srls(self.pts.edm, self.pts.points, indices=[self.index],
                                W=np.ones(self.pts.edm.shape))

    def test_multiple_weights(self):
        print('TestSRLS:test_multiple')
        for i in range(100):
            self.create_points()
            self.zero_weights(0.0)
            self.zero_weights(0.1)
            self.zero_weights(1.0)

    def test_srls_rescale():

        anchors = np.array([[ 0.        ,  8.44226166,  0.29734295],
                            [ 1.        ,  7.47840264,  1.41311759],
                            [ 2.        ,  8.08093318,  2.21959719],
                            [ 3.        ,  4.55126532,  0.0456345 ],
                            [ 4.        ,  5.10971446, -0.01223217],
                            [ 5.        ,  2.95745961, -0.77572604],
                            [ 6.        ,  3.12145804,  0.80297295],
                            [ 7.        ,  2.29152331, -0.48021431],
                            [ 8.        ,  1.53137609, -0.03621697],
                            [ 9.        ,  0.762208  ,  0.70329037]])
        W = np.ones(anchors.shape[0])
        x = np.ones(anchors.shape[1]) * 4.
        r2 = np.linalg.norm(anchors - x[None,:], axis=1)**2
        sigma = 3.

        # Normal ranging
        x_srls = SRLS(anchors, W, r2)
        assert np.allclose(x, x_srls)

        # Rescaled ranging
        x_srls_resc = SRLS(anchors, W, sigma * r2, rescale=True)
        assert np.allclose(x, x_srls_resc)

    def zero_weights(self, noise=0.1):
        print('TestSRLS:test_zero_weights({})'.format(noise))
        other = np.delete(range(self.pts.N), self.index)
        edm_noisy = create_noisy_edm(self.pts.edm, noise)

        # missing anchors
        N_missing = 2
        indices = np.random.choice(other, size=N_missing, replace=False)
        reduced_points = np.delete(self.pts.points, indices, axis=0)
        points_missing = create_from_points(reduced_points, PointSet)
        edm_anchors = np.delete(edm_noisy, indices, axis=0)
        edm_anchors = np.delete(edm_anchors, indices, axis=1)
        missing_anchors = reconstruct_srls(
            edm_anchors, points_missing.points, indices=[self.index], W=None,
            print_out=False)

        # missing distances
        weights = np.ones(edm_noisy.shape)
        weights[indices, self.index] = 0.0
        weights[self.index, indices] = 0.0

        missing_distances = reconstruct_srls(
            edm_noisy, self.pts.points, indices=[self.index], W=weights)
        left_distances = np.delete(range(self.pts.N), indices)

        self.assertTrue(np.linalg.norm(
            missing_distances[left_distances, :] - missing_anchors) < 1e-10, 'anchors moved.')
        self.assertTrue(np.linalg.norm(
            missing_distances[self.index, :] - missing_anchors[self.index, :]) < 1e-10, 'point moved.')
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
