#!/usr/bin/env python
# module TEST_SRLS

import unittest
import numpy as np
import sys
sys.path.append('./test/')

from test_common import BaseCommon

from pylocus.point_set import PointSet, create_from_points
from pylocus.simulation import create_noisy_edm
from pylocus.algorithms import reconstruct_srls
from pylocus.lateration import SRLS, get_lateration_parameters, GeometryError

class TestSRLS(BaseCommon.TestAlgorithms):
    def setUp(self):
        BaseCommon.TestAlgorithms.setUp(self)
        self.create_points()
        # for d=3, N_missing=2, need at least 6 anchors (no rescale) or 7 anchors (rescaled).
        self.N_relaxed = range(6, 10)
        self.methods = ['normal', 'rescale', 'fixed']
        self.eps = 1e-8

    def create_points(self, N=10, d=2):
        self.pts = PointSet(N, d)
        self.pts.set_points('random')
        self.pts.init()
        self.n = 1

    def call_method(self, method=''):
        print('TestSRLS:call_method')
        if method == '' or method == 'normal':
            return reconstruct_srls(self.pts.edm, self.pts.points, 
                                    W=np.ones(self.pts.edm.shape))
        elif method == 'rescale':
            return reconstruct_srls(self.pts.edm, self.pts.points, 
                                    W=np.ones(self.pts.edm.shape), rescale=True)
        elif method == 'fixed' and self.pts.d == 3:
            return reconstruct_srls(self.pts.edm, self.pts.points, 
                                    W=np.ones(self.pts.edm.shape), rescale=False,
                                    z=self.pts.points[0, 2])

    def test_fail(self):
        # Example point set that used to fail. With the newer SRLS version this is fixed. 
        # Status:  July 16, 2018
        points_fail = np.array([[0.00250654, 0.89508715, 0.35528746],
                                [0.52509683, 0.88692205, 0.76633946],
                                [0.64764605, 0.94040708, 0.20720253],
                                [0.69637586, 0.99566993, 0.49537693],
                                [0.64455557, 0.46856155, 0.80050257],
                                [0.90556836, 0.75831552, 0.81982037],
                                [0.86634135, 0.5139182 , 0.14738743],
                                [0.29145628, 0.54500108, 0.6586396 ]])
        method = 'rescale'
        self.pts.set_points(points=points_fail)
        points_estimate = self.call_method(method=method)
        error = np.linalg.norm(self.pts.points - points_estimate) 
        self.assertTrue(error < self.eps, 
                        'error: {} not smaller than {}'.format(error, self.eps))
        
        points_fail = np.array([[0.63, 0.45],
                                [0.35, 0.37],
                                [0.69, 0.71],
                                [0.71, 0.73],
                                [0.43, 0.44],
                                [0.58, 0.59]])
        method = 'normal'

        self.pts.set_points(points=points_fail)
        points_estimate = self.call_method(method=method)
        error = np.linalg.norm(self.pts.points - points_estimate) 
        self.assertTrue(error < self.eps, 
                         'error: {} not smaller than {}'.format(error, self.eps))

    def test_multiple_weights(self):
        print('TestSRLS:test_multiple_weights')
        for i in range(self.n_it):
            self.create_points()
            self.zero_weights(0.0)
            self.zero_weights(0.1)
            self.zero_weights(1.0)

    def test_srls_rescale(self):
        print('TestSRLS:test_srls_rescale')
        anchors = np.array([[0.,  8.44226166,  0.29734295],
                            [1.,  7.47840264,  1.41311759],
                            [2.,  8.08093318,  2.21959719],
                            [3.,  4.55126532,  0.0456345],
                            [4.,  5.10971446, -0.01223217],
                            [5.,  2.95745961, -0.77572604],
                            [6.,  3.12145804,  0.80297295],
                            [7.,  2.29152331, -0.48021431],
                            [8.,  1.53137609, -0.03621697],
                            [9.,  0.762208,  0.70329037]])
        sigma = 3.

        N, d = anchors.shape
        w = np.ones((N, 1))
        x = np.ones(d) * 4.
        r2 = np.linalg.norm(anchors - x[None, :], axis=1)**2
        r2.resize((len(r2), 1))

        # Normal ranging
        x_srls = SRLS(anchors, w, r2)
        np.testing.assert_allclose(x, x_srls)

        # Rescaled ranging
        x_srls_resc, scale = SRLS(anchors, w, sigma * r2, rescale=True)
        self.assertLess(abs(1/scale - sigma), self.eps, 'not equal: {}, {}'.format(scale, sigma))
        np.testing.assert_allclose(x, x_srls_resc)

    def test_srls_fixed(self):
        print('TestSRLS:test_srls_fixed')
        self.create_points(N=10, d=3)
        zreal = self.pts.points[0, 2]
        xhat = reconstruct_srls(self.pts.edm, self.pts.points,  
                                W=np.ones(self.pts.edm.shape), rescale=False, 
                                z=zreal)
        if xhat is not None:
            np.testing.assert_allclose(xhat[0, 2], zreal)
            np.testing.assert_allclose(xhat, self.pts.points)

    def test_srls_fail(self):
        anchors = np.array([[11.881,  3.722,  1.5  ],
                            [11.881,  14.85,   1.5 ],
                            [11.881,  7.683,  1.5  ]])
        w = np.ones((3, 1))
        distances = [153.32125426, 503.96654466, 234.80741129] 
        z = 1.37
        self.assertRaises(GeometryError, SRLS, anchors, w, distances, False, z)

    def zero_weights(self, noise=0.1):
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
            edm_anchors, points_missing.points, W=None,
            print_out=False)

        # missing distances
        weights = np.ones(edm_noisy.shape)
        weights[indices, index] = 0.0
        weights[index, indices] = 0.0

        missing_distances = reconstruct_srls(
            edm_noisy, self.pts.points, W=weights)
        left_distances = np.delete(range(self.pts.N), indices)

        self.assertTrue(np.linalg.norm(
            missing_distances[left_distances, :] - missing_anchors) < self.eps, 'anchors moved.')
        self.assertTrue(np.linalg.norm(
            missing_distances[index, :] - missing_anchors[index, :]) < self.eps, 'point moved.')
        if noise == 0.0:
            error = np.linalg.norm(missing_anchors - points_missing.points)
            u, s, v = np.linalg.svd(self.pts.points[other, :])
            try:
                self.assertTrue(error < self.eps, 'error {}'.format(error))
            except:
                print('failed with anchors (this can be due to unlucky geometry):',
                      points_missing.points)
                raise


if __name__ == "__main__":
    unittest.main()
