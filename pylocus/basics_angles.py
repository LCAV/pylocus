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
