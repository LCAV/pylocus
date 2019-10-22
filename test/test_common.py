#!/usr/bin/env python
# module TEST_COMMON

import unittest
import numpy as np
from abc import ABC, abstractmethod


class BaseCommon:
    ''' Empty class such that TestCommon is not run as a test class.
    TestAlgorithms is abstract and needs to be run through the inheriting classes.
    '''

    class TestAlgorithms(ABC, unittest.TestCase):
        def setUp(self):
            self.eps = 1e-10
            self.success_rate = 50
            self.n_it = 100
            self.N_zero = range(8, 12)
            self.N_relaxed = range(4, 10)
            self.methods = ['']

        @abstractmethod
        def create_points(self, N, d):
            raise NotImplementedError(
                'Call to virtual method! create_points has to be defined by sublcasses!')

        @abstractmethod
        def call_method(self, method=''):
            raise NotImplementedError(
                'Call to virtual method! call_method has to be defined by sublcasses!')

        def test_multiple(self):
            print('TestCommon:test_multiple')
            for i in range(self.n_it): # seed 381 used to fail.
                np.random.seed(i)
                self.test_zero_noise(it=i)

        def test_zero_noise(self, it=0):
            print('TestCommon:test_zero_noise')
            for N in self.N_zero:
                for d in (2, 3):
                    self.create_points(N, d) 
                    for method in self.methods: 
                        points_estimate = self.call_method(method)
                        if points_estimate is None:
                            continue
                        np.testing.assert_allclose(self.pts.points, points_estimate, self.eps)
