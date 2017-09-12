#!/usr/bin/env python
# module TEST_EMDS

import unittest
import numpy as np
from test_common import BaseCommon

from pylocus.point_set import HeterogenousSet
from pylocus.algorithms import reconstruct_emds


class TestEMDS(BaseCommon.TestAlgorithms):
    def create_points(self, N=10, d=3):
        print('TestEMDS:setUp')
        self.pts = HeterogenousSet(N, d)
        self.pts.set_points('normal')

    def call_method(self):
        print('TestEMDS:call_method')
        return reconstruct_emds(self.pts.edm, Om=self.pts.Om,
                                real_points=self.pts.points)


if __name__ == "__main__":
    unittest.main()
