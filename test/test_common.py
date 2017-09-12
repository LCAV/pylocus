#!/usr/bin/env python
# module TEST_COMMON

import unittest
import numpy as np
from abc import *


class BaseCommon:
    ''' Empty class such that TestCommon is not run as a test class.
    BaseCommon is abstract and needs to be run through the inheriting classes.
    '''

    class TestAlgorithms(ABC, unittest.TestCase):

        @abstractmethod
        def create_points(self, N, d):
            raise NotImplementedError(
                'Call to virtual method! create_points has to be defined by sublcasses!')

        @abstractmethod
        def call_method(self):
            raise NotImplementedError(
                'Call to virtual method! call_method has to be defined by sublcasses!')

        def test_multiple(self):
            print('TestCommon:test_multiple')
            for i in range(3):
                self.test_zero_noise()

        def test_zero_noise(self):
            print('TestCommon:test_zero_noise')
            for N in range(4, 10):
                for d in (2, 3):
                    self.create_points(N, d)
                    points_estimate = self.call_method()
                    error = np.linalg.norm(self.pts.points - points_estimate) 
                    self.assertTrue(error < 1e-10, 'noiseless case did not give 0 error:{}'.format(error))
