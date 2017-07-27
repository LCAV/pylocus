#!/usr/bin/env python
# module TEST_SRLS

import unittest
import numpy as np
import logging
from .point_configuration import PointSet, create_from_points


class TestSRLS(unittest.TestCase):

    def setUp(self):
        N = 10
        d = 3
        self.pts = PointSet(N, d)
        self.pts.set_points('random')

    def test_multiple(self):
        for i in range(100):
            self.setUp()
            self.test_zero_weights(-1, 0.1)
            self.test_zero_weights(-1, 0.0)

    def test_zero_weights(self, index=-1, noise=0.1):
        from .basics import create_noisy_edm
        from .algorithms import reconstruct_srls
        self.pts.set_points(mode='random')
        other = np.delete(range(self.pts.N), index)
        edm_noisy = create_noisy_edm(self.pts.edm, noise)

        # missing anchors
        N_missing = 3
        indices = np.random.choice(other, size=N_missing, replace=False)
        reduced_points = np.delete(self.pts.points, indices, axis=0)
        points_missing = create_from_points(reduced_points, PointSet)
        edm_missing = np.delete(edm_noisy, indices, axis=0)
        edm_missing = np.delete(edm_missing, indices, axis=1)
        missing_anchors = reconstruct_srls(
            edm_missing, points_missing.points, index=index, weights=None)

        # missing distances
        weights = np.ones(edm_noisy.shape)
        weights[indices, :] = 0.0
        weights[:, indices] = 0.0
        missing_distances = reconstruct_srls(
            edm_noisy, self.pts.points, index=index, weights=weights)
        left_distances = np.delete(range(self.pts.N), indices)

        self.assertTrue(np.linalg.norm(missing_distances[
                        left_distances, :] - missing_anchors) < 1e-10, 'anchors moved.')
        self.assertTrue(np.linalg.norm(
            missing_distances[index, :] - missing_anchors[index, :]) < 1e-10, 'point moved.')
        if noise == 0.0:
            print('error',np.linalg.norm(missing_distances - self.pts.points))
            self.assertTrue(np.linalg.norm(
                missing_distances - self.pts.points) < 1e-10, 'error')

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TestMDS.test_angle_MDS").setLevel(logging.DEBUG)
    unittest.main()
