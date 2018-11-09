#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pylocus.point_set import PointSet
from pylocus.algorithms import reconstruct_srls
from pylocus.simulation import create_noisy_edm, create_mask, create_weights
from pylocus.basics import mse, rmse

points = PointSet(N=5, d=2)
points.set_points('random')
print("point to localize:", points.points[0, :])
print("anchors:", points.points[1:, :])

std = 0.1
edm_noisy = create_noisy_edm(points.edm, noise=std)

mask = create_mask(points.N , method='none')
weights = create_weights(points.N, method='one')
weights = np.multiply(mask, weights)

points_estimated  = reconstruct_srls(edm_noisy, points.points, W=weights)
error = mse(points_estimated[0, :], points.points[0, :])

print("estimated point: {}, original point: {}, mse: {:2.2e}".format(points_estimated[0, :],
                                                                points.points[0, :],
                                                                error))
