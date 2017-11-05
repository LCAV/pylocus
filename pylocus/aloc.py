#!/usr/bin/env python
# module A-LOC
from algorithm_constraints import get_all_constraints, projection
from basics_alternating import get_absolute_angles_tensor
from basics_aloc import from_0_to_2pi, get_point, get_inner_angle, get_absolute_angle, get_theta_tensor
from math import pi, floor, cos, sin
from plots import plot_theta_errors, plot_point_sets
from .basics import rmse
import numpy as np


def reconstruct(Pi, Pj, i, j, theta_tensor, Pk='', k=-1, print_out=False):
    '''
    Returns the reconstruction of pointset from inner angles vector theta. If Pk and k are set, it corrects for the original flip.
    Args:
        Pi: point i of base line.
        Pj: point j of base line.
        i: index of point Pi.
        j: index of point Pj.
        theta: tensor containing inner angles
        Pk: coordinates of point k (cannot be 0 for now.)
        k: index of point Pk
    Returns:
        own: obtained AngleSet object
    '''
    from .algorithms import procrustes, apply_transform
    from .point_set import create_from_points, AngleSet
    from .basics_angles import get_inner_angle

    def intersect(Pi, Pj, alpha_ik, alpha_jk):
        A = np.matrix([[cos(alpha_ik), -cos(alpha_jk)],
                       [sin(alpha_ik), -sin(alpha_jk)]])
        s = np.linalg.solve(A, Pj)
        P2 = s[0] * np.array([cos(alpha_ik), sin(alpha_ik)])
        P2_same = Pj + s[1] * np.array([cos(alpha_jk), sin(alpha_jk)])
        assert np.isclose(P2, P2_same).all()

        theta = get_inner_angle(P0, [P1, P2])

        # TODO not robust to multiples of pi.
        #assert abs(theta - theta_tensor[i, j, l]) < 1e-10, '{}-{} not smaller than 1e-10'.format(
        #    degrees(theta), degrees(theta_tensor[i, j, l]))
        theta = get_inner_angle(P1, [P0, P2])
        #assert abs(theta - theta_tensor[j, i, l]) < 1e-10, '{}-{} not smaller than 1e-10'.format(
        #    degrees(theta), degrees(theta_tensor[j, i, l]))
        return P2

    # Initialize to fix orientation
    N = theta_tensor.shape[0]
    
    points = np.zeros((N, Pi.shape[0]))

    P0 = np.array((0, 0))
    P1 = np.array((Pj[0] - Pi[0], 0))

    points[i, :] = P0
    points[j, :] = P1
    left_indices = np.delete(range(N), (i, j))

    first = True
    for counter, l in enumerate(left_indices):
        alpha_il = theta_tensor[i, j, l]
        alpha_jl = pi - theta_tensor[j, i, l]
        Pl = intersect(P0, P1, alpha_il, alpha_jl)
        if not first:
            try:
                previous = left_indices[counter - 1]
                estimated = get_inner_angle(points[previous, :], (P0, Pl))
                expected = theta_tensor[previous, i, l]
                assert abs(
                    estimated - expected) < 0.1, 'error {}'.format(abs(estimated - expected))
            except:
                alpha_il = -alpha_il
                alpha_jl = -alpha_jl
                Pl = intersect(P0, P1, alpha_il, alpha_jl)
        points[l, :] = Pl
        first = False
    anchors = np.r_[Pi,Pj,Pk].reshape(-1,2)
    __, R, t, c = procrustes(anchors, points[[i,j,k],:], scale=False)
    fitted = apply_transform(R, t, c, points, anchors)
    own = create_from_points(fitted, AngleSet)
    return own


def degrees(angle):
    return 180 * angle / pi
