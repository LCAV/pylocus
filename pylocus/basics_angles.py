#!/usr/bin/env python
# module BASICS_ANGLES
from math import atan2, pi
import numpy as np


def change_angles(method, theta, tol=1e-10):
    """ Function used by all angle conversion functions (from_x_to_x_pi(...))"""
    try:
        theta_new = np.zeros(theta.shape)
        for i, thet in enumerate(theta):
            try:
                # theta is vector
                theta_new[i] = method(thet, tol)
            except:
                # theta is matrix
                for j, th in enumerate(thet):
                    try:
                        theta_new[i, j] = method(th, tol)
                    except:
                        # theta is tensor
                        for k, t in enumerate(th):
                            theta_new[i, j, k] = method(t, tol)
        return theta_new
    except:
        return method(theta, tol)


def from_0_to_pi(theta):
    def from_0_to_pi_scalar(theta, tol):
        theta = from_0_to_2pi(theta)
        theta = min(theta, 2 * pi - theta)
        assert theta >= 0 and theta <= pi, "{} not in [0, pi]".format(theta)
        return theta
    return change_angles(from_0_to_pi_scalar, theta)


def from_0_to_2pi(theta, tol=1e-10):
    def from_0_to_2pi_scalar(theta, tol):
        theta = theta % (2 * pi)
        # eliminate numerical issues of % function
        if abs(theta - 2 * pi) < tol:
            theta = theta - 2 * pi
        if theta < 0:
            theta = 2 * pi + theta
        assert theta >= 0 and theta <= 2 * \
            pi, "{} not in [0, 2pi]".format(theta)
        return theta
    return change_angles(from_0_to_2pi_scalar, theta, tol)


def get_absolute_angle(Pi, Pj):
    if (Pi == Pj).all():
        return 0
    y = Pj[1] - Pi[1]
    x = Pj[0] - Pi[0]
    theta_ij = atan2(y, x)
    return from_0_to_2pi(theta_ij)


def get_absolute_angles_tensor(thetas, thetas_noisy, ks, N, tol=1e-10, print_out=False):
    '''
    Args:
        thetas:         Tensor of inner angles from point k to other points.  (without noise)
        thetas_noisy:   Same as thetas, but with noise.
        k:              Point for which alpha is evaluated.
        N:              Number of points.
    Returns:
    
    '''
    def get_alpha(alphas_k, corners_k, corner):
        idx_old = [idx for idx, tria in enumerate(corners_k) if tria == corner]
        assert len(idx_old) == 1, "idx_old:{}".format(idx_old)
        idx_old = idx_old[0]
        return alphas_k[idx_old]
    corners_k = []
    alphas_k = []
    alphas_tensor = np.zeros(thetas.shape)
    for k in ks:
        other_indices = np.delete(range(N), k)
        first = other_indices[0]
        left_indices = other_indices[1:]

        # fix direction of first angle. (alpha_01)
        second = left_indices[0]
        corner_first = (k, first, second)
        theta_01 = thetas[corner_first]
        theta_01_noisy = thetas_noisy[corner_first]
        alpha_01 = theta_01
        alpha_01_noisy = theta_01_noisy
        alphas_k.append(alpha_01_noisy)
        corners_k.append(corner_first)
        alphas_tensor[corner_first] = alpha_01_noisy
        alphas_tensor[k, second, first] = -alpha_01_noisy
        if (print_out):
            print('first: alpha_{}{}={}'.format(first, second, alphas_k[-1]))

        # find directions of other lines accordingly (alpha_0i)
        left_indices = other_indices[2:]
        for idx, i in enumerate(left_indices):
            corner12 = (k, second, i)
            theta_12 = thetas[corner12]
            theta_12_noisy = thetas_noisy[corner12]
            corner02 = (k, first, i)
            theta_02 = thetas[corner02]
            theta_02_noisy = thetas_noisy[corner02]
            corner01 = (k, first, second)
            theta_01 = thetas[corner01]
            theta_01_noisy = thetas_noisy[corner01]
            if (print_out):
                print('     theta_{}{}={}'.format(first, second, theta_01))
                print('     theta_{}{}={}'.format(first, i, theta_02))
                print('     theta_{}{}={}'.format(second, i, theta_12))
            if abs(from_0_to_pi(theta_01 + theta_02) - theta_12) < tol:
                alpha_02_noisy = -theta_02_noisy
                alpha_02 = -theta_02
            else:
                alpha_02_noisy = theta_02_noisy
                alpha_02 = -theta_02
            alphas_k.append(alpha_02_noisy)
            corners_k.append(corner02)
            alphas_tensor[corner02] = alpha_02_noisy
            alphas_tensor[k, i, first] = -alpha_02_noisy
            if (print_out):
                print('second alpha_{}{}={}'.format(first, i, alpha_02_noisy))

        # find directions between all lines (alpha_ij)
        left_indices = other_indices[1:]
        for idx, i in enumerate(left_indices):
            for j in left_indices[idx + 1:]:
                corner0i = (k, first, i)
                alpha_0i = get_alpha(alphas_k, corners_k, corner0i)
                corner0j = (k, first, j)
                alpha_0j = get_alpha(alphas_k, corners_k, corner0j)
                alpha_ij = alpha_0j - alpha_0i
                cornerij = (k, i, j)
                theta_ij_noisy = thetas_noisy[cornerij]
                if abs(alpha_ij) > pi:
                    if alpha_ij > 0:
                        alpha_ij_noisy = 2 * pi - theta_ij_noisy
                    else:
                        alpha_ij_noisy = theta_ij_noisy - 2 * pi
                else:
                    if alpha_ij > 0:
                        alpha_ij_noisy = theta_ij_noisy
                    else:
                        alpha_ij_noisy = -theta_ij_noisy
                alphas_k.append(alpha_ij_noisy)
                corners_k.append(cornerij)
                alphas_tensor[cornerij] = alpha_ij_noisy
                alphas_tensor[k, j, i] = -alpha_ij_noisy
    alphas_k = [from_0_to_2pi(alph) for alph in alphas_k]
    corners_k = np.array(corners_k).reshape((-1, 3))
    alphas_ordered = alphas_k
    alphas_tensor = from_0_to_2pi(alphas_tensor)
    return alphas_ordered, alphas_tensor, corners_k


def get_inner_angle(Pk, Pij):
    theta_ki = get_absolute_angle(Pk, Pij[0])
    theta_kj = get_absolute_angle(Pk, Pij[1])
    theta = abs(theta_ki - theta_kj)
    return from_0_to_pi(theta)


def rmse_2pi(x, xhat):
    ''' Calcualte rmse between vector or matrix x and xhat, ignoring mod of 2pi.'''
    real_diff = from_0_to_pi(x - xhat)
    np.square(real_diff, out=real_diff)
    sum_ = np.sum(real_diff)
    return sqrt(sum_ / len(x))


def get_point(theta_ik, theta_jk, Pi, Pj):
    """ Calculate coordinates of point Pk given two points Pi, Pj and inner angles.  :param theta_ik: Inner angle at Pi to Pk.
    :param theta_jk: Inner angle at Pj to Pk.
    :param Pi: Coordinates of point Pi.
    :param Pj: Coordinates of point Pj.

    :return: Coordinate of point Pk.
    """
    A = np.array([[sin(theta_ik), -cos(theta_ik)],
                  [sin(theta_jk), -cos(theta_jk)]])
    B = np.array([[sin(theta_ik), -cos(theta_ik), 0, 0],
                  [0, 0, sin(theta_jk), -cos(theta_jk)]])
    p = np.r_[Pi, Pj]
    Pk = np.linalg.solve(A, np.dot(B, p))
    return Pk


def get_theta_tensor(theta, corners, N):
    theta_tensor = np.zeros([N, N, N])
    for k, idx in enumerate(corners):
        theta_tensor[int(idx[0]), int(idx[1]), int(idx[2])] = theta[k]
        theta_tensor[int(idx[0]), int(idx[2]), int(idx[1])] = theta[k]
    return theta_tensor


def get_index(corners, Pk, Pij):
    ''' get index mask corresponding to angle at corner Pk with Pi, Pj.'''
    angle1 = [Pk, Pij[0], Pij[1]]
    angle2 = [Pk, Pij[1], Pij[0]]
    index = np.bitwise_or(corners == angle1, corners == angle2)
    return index.all(axis=1)
