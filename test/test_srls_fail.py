#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Frederike Duembgen <frederike.duembgen@gmail.com>
#
# Distributed under terms of the MIT license.

"""
test_srls_fail.py: 
"""
import unittest
import numpy as np
from .test_common import BaseCommon

from pylocus.algorithms import reconstruct_srls
from pylocus.basics import get_edm

class TestSRLSFail(unittest.TestCase):
    """ trying to reproduce a fail of SRLS. """ 
    def setUp(self):
        # from error logs: 
        self.all_points = np.array([[0.89, 2.87, 1.22], 
                                     [2.12000000001, 0.0, 1.5], 
                                     [5.83999999999, 2.5499998, 1.37999999], 
                                     [5.83999999999, 4.6399997, 1.57000001], 
                                     [0.100000000001, 3.0, 1.48], 
                                     [0.100000000001, 4.20000000002, 1.5], 
                                     [2.879999999999, 3.33999999999, 0.75], 
                                     [5.759999999998, 3.46, 1.0], 
                                     [3.54, 6.8700001, 1.139999]]) 
        self.anchors = self.all_points[1:, :] 
        self.xhat = [0.8947106, 2.87644499, 1.22189597]

        distances = [1.1530589508606381, 1.6797066823110891, 1.1658051261713582, 0.42021624764667165, 1.1196561208517539, 0.44373006779135082, 1.3051970883541162, 2.6628083049963136]
        [M, d] = self.anchors.shape
        N = M + 1
        self.edm = np.zeros((N, N))
        self.edm[1:, 1:] = get_edm(self.anchors)
        self.edm[0, 1:] = np.array(distances) ** 2.0
        self.edm[1:, 0] = np.array(distances) ** 2.0

        weights = [10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0]
        self.weights = np.zeros((N, N))
        self.weights[1:, 1:] = 10.0
        self.weights[0, 1:] = weights
        self.weights[1:, 0] = weights

    def test_strange_case(self):
        xhat = reconstruct_srls(self.edm, self.all_points, 
                                W=self.weights)
        print(xhat[0])
        xhat = reconstruct_srls(self.edm, self.all_points, 
                                W=self.weights, rescale=False,
                                z=self.all_points[0, 2])
        print(xhat[0])

if __name__ == "__main__":
    unittest.main()
