#!/usr/bin/env python
# module A-LOC
from algorithm_constraints import get_all_constraints, projection
from basics_alternating import get_absolute_angles_tensor
from basics_aloc import from_0_to_2pi, get_point, get_inner_angle, get_absolute_angle, get_theta_tensor
from math import pi, floor, cos, sin
from pylocus.point_set import AngleSet
from plots import plot_theta_errors, plot_point_sets
from .basics import rmse
import numpy as np


def reconstruct_from_inner_angles(P0, P1, theta_02, theta_12, theta):
    '''
    Returns the reconstruction of pointset from inner angles vector theta.
    Args:
        P0: point 0
        P1: point 1
        theta_02: absolute orientation of line between P0 and P2
        theta_12: absolute orientation of line between P1 and P2
        theta: vector containing inner angles
    Returns:
        own: obtained AngleSet object
    '''
    N = theta.shape[0]
    d = P0.shape[0]
    show = False
    own = AngleSet(N, d)
    if (show):
        print('N =  %r, d = %r' % (N, d))
    # Initialize to fix orientation
    own.points[0, :] = P0
    own.points[1, :] = P1
    theta_01 = get_absolute_angle(P0, P1)
    theta_10 = get_absolute_angle(P1, P0)
    own.abs_angles[0, 2] = theta_02
    own.abs_angles[2, 0] = from_0_to_2pi(theta_02 + pi)
    own.abs_angles[1, 2] = theta_12
    own.abs_angles[2, 1] = from_0_to_2pi(theta_12 + pi)
    own.points[2, :] = get_point(own.abs_angles[0, 2], own.abs_angles[
                                 1, 2], own.points[0, :], own.points[1, :])

    # Calculate point coordinates
    if N <= 3:
        return own
    for k in range(3, N):
        i = k - 2  # 1
        j = k - 1  # 2
        m = k - 3  # 0
        Pi = own.points[i, :]
        Pj = own.points[j, :]
        Pm = own.points[m, :]

        theta_ij = own.abs_angles[i, j]
        theta_ji = own.abs_angles[j, i]
        thetai_jk = theta[i, j, k]
        thetaj_ik = theta[j, i, k]
        if (show):
            print('i,j,k', i, j, k)
            print('theta_ij', theta_ij)
            print('thetai_jk', thetai_jk)
            print('thetaj_ik', thetaj_ik)

        # try case one.
        theta_ik1 = from_0_to_2pi(theta_ij + thetai_jk)
        theta_jk1 = from_0_to_2pi(theta_ji - thetaj_ik)
        Pk1 = get_point(theta_ik1, theta_jk1, Pi, Pj)
        test_thetak_mi = get_inner_angle(Pk1, (Pm, Pi))
        truth_thetak_mi = theta[k, m, i]
        error1 = abs(test_thetak_mi - truth_thetak_mi)
        if error1 > 1e-10:
            # try case two.
            if (show):
                print('wrong choice, thetak_mi should be', truth_thetak_mi, 'is',
                      test_thetak_mi, 'diff:', truth_thetak_mi - test_thetak_mi)
            theta_ik2 = from_0_to_2pi(theta_ij - thetai_jk)
            theta_jk2 = from_0_to_2pi(theta_ji + thetaj_ik)
            Pk2 = get_point(theta_ik2, theta_jk2, Pi, Pj)
            test_thetak_mi = get_inner_angle(Pk2, (Pm, Pi))
            truth_thetak_mi = theta[k, m, i]
            error2 = abs(test_thetak_mi - truth_thetak_mi)
            if abs(test_thetak_mi - truth_thetak_mi) > 1e-10:
                # If both errors are not so great, choose the better one.
                # print('new thetak_mi should be     :%4.f, is: %4.f, diff: %r' % (truth_thetak_mi_new, test_thetak_mi_new, truth_thetak_mi_new - test_thetak_mi_new))
                # print('previous thetak_mi should be:%4.f, is: %4.f, diff: %r' % (truth_thetak_mi, test_thetak_mi, truth_thetak_mi - test_thetak_mi))
                if error1 > error2:
                    #if error2 > 1e-3:
                    #    print('none of the options gives angle < 1e-3 (%r,%r)' % (error1, error2))
                    #else:
                    theta_ik = theta_ik2
                    theta_jk = theta_jk2
                    Pk = Pk2
                else:
                    #if error1 > 1e-3:
                    #    print('none of the options gives angle < 1e-3 (%r,%r)' % (error1, error2))
                    #else:
                    theta_ik = theta_ik1
                    theta_jk = theta_jk1
                    Pk = Pk1
            else:
                theta_ik = theta_ik2
                theta_jk = theta_jk2
                Pk = Pk2
        else:
            theta_ik = theta_ik1
            theta_jk = theta_jk1
            Pk = Pk1

        own.points[k, :] = Pk
        own.abs_angles[i, k] = from_0_to_2pi(theta_ik)
        own.abs_angles[j, k] = from_0_to_2pi(theta_jk)
        own.abs_angles[k, i] = from_0_to_2pi(theta_ik + pi)
        own.abs_angles[k, j] = from_0_to_2pi(theta_jk + pi)
        #own.plot_some(range(m,k+1),'some')
    own.init()
    own.get_theta_tensor()
    return own


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

    def intersect(Pi, Pj, alpha_ik, alpha_jk):
        A = np.matrix([[cos(alpha_ik), -cos(alpha_jk)],
                       [sin(alpha_ik), -sin(alpha_jk)]])
        b = Pj - Pi
        s = np.linalg.solve(A, b) 
        Pk_1 = Pi + s[0] * np.array([cos(alpha_ik), sin(alpha_ik)])
        Pk_2 = Pj + s[1] * np.array([cos(alpha_jk), sin(alpha_jk)])
        assert np.isclose(Pk_1, Pk_2).all()
        return Pk_1

    # Initialize to fix orientation
    N = theta_tensor.shape[0]
    own = AngleSet(N, Pi.shape[0])
    own.points[i, :] = Pi
    own.points[j, :] = Pj
    alpha_ij = get_absolute_angle(Pi, Pj)
    alpha_ji = get_absolute_angle(Pj, Pi)
    if (print_out):
        print('alpha_ij, alpha_ji', 180 * np.array((alpha_ij, alpha_ji)) / pi)

    left_indices = np.delete(np.arange(N), (i, j))

    # Get directed angles for both points.
    __, alphas_tensor, __ = get_absolute_angles_tensor(
        theta_tensor, theta_tensor, (i, j), N)

    # Correct for consistent flip.
    if abs(alphas_tensor[j, j, k] - alpha_ji) < pi:
        if (print_out):
            print('flip', j)
        alphas_tensor[j, :, :] *= -1
    if abs(alphas_tensor[i, i, k] - alpha_ij) < pi:
        if (print_out):
            print('flip', i)
        alphas_tensor[i, :, :] *= -1
    if (print_out):
        print('alphas_tensor', alphas_tensor * 180 / pi)

    # Correct for true flip.
    alpha_ij = get_absolute_angle(Pi, Pj)
    alphas_tensor[i, i, :] = from_0_to_2pi(alphas_tensor[i, j, :] + alpha_ij)
    if k >= 0:
        alpha_ik = get_absolute_angle(Pi, Pk)
        calc_alpha_ik = alphas_tensor[i, i, k]
        diff_calc = from_0_to_2pi(alpha_ij - calc_alpha_ik)
        diff = from_0_to_2pi(alpha_ij - alpha_ik)
        if diff_calc < pi and diff_calc > 0:
            if not (diff < pi and diff > 0):
                alphas_tensor *= -1.0
        else:
            if not (diff > pi or diff < 0):
                alphas_tensor *= -1.0

    alpha_ji = get_absolute_angle(Pj, Pi)
    alphas_tensor[i, i, :] = from_0_to_2pi(alphas_tensor[i, j, :] + alpha_ij)
    alphas_tensor[:, range(N), range(N)] = 0.0
    alphas_tensor[j, j, :] = from_0_to_2pi(alphas_tensor[j, i, :] + alpha_ji)
    alphas_tensor[:, range(N), range(N)] = 0.0
    alphas_tensor[i, :, i] = from_0_to_2pi(alphas_tensor[i, :, j] + alpha_ji)
    alphas_tensor[:, range(N), range(N)] = 0.0
    alphas_tensor[j, :, j] = from_0_to_2pi(alphas_tensor[j, :, i] + alpha_ij)
    alphas_tensor[:, range(N), range(N)] = 0.0

    for k in left_indices:
        alpha_ik = alphas_tensor[i, i, k]
        alpha_jk = alphas_tensor[j, j, k]
        if print_out and k >= 0:
            real_alpha_ik = get_absolute_angle(Pi, Pk)
            real_alpha_jk = get_absolute_angle(Pj, Pk)
            print('real alpha_ik, alpha_jk', 180 *
                  np.array((real_alpha_ik, real_alpha_jk)) / pi)
            print('calc alpha_ik, alpha_jk', 180 *
                  np.array((alpha_ik, alpha_jk)) / pi)
        Pk_1 = intersect(Pi, Pj, alpha_ik, alpha_jk)
        own.points[k, :] = Pk_1
    own.init()
    own.get_theta_tensor()
    return own

