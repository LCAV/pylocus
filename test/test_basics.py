#!/usr/bin/env python
# module TEST_BASICS

import unittest
import numpy as np
from pylocus.basics import projection, mse


class TestProjection(unittest.TestCase):

    def setUp(self, N=4):
        self.N = N
        self.A = np.random.rand(self.N, self.N)
        self.b = np.random.rand(self.N)
        self.x = np.random.rand(self.N, self.N)

    def test_multiple(self):
        for i in range(10):
            self.test_square_matrix()
            self.test_square_vector()

    def test_square_vector(self):
        self.A = np.random.rand(self.N, self.N)
        self.b = np.random.rand(self.N)
        self.x = np.random.rand(self.N)
        self.test_projection()

    def test_square_matrix(self):
        self.A = np.random.rand(self.N, self.N)
        self.b = np.random.rand(self.N)
        self.x = np.random.rand(self.N, self.N)
        self.test_projection()

    def test_projection(self):
        xhat, cost, constraints = projection(self.x, self.A, self.b)
        self.assertTrue(constraints <= 1e-15, 'Constraints not satisfied:{}'.format(constraints))

        xhat2, __, __ = projection(xhat, self.A, self.b)
        idempotent_error = mse(xhat, xhat2)
        self.assertTrue(idempotent_error < 1e-10, 'Projection not idempotent.')


if __name__ == "__main__":
    unittest.main()
