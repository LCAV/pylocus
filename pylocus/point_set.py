#!/usr/bin/env python
""" Module containing classes for handling 2D or 3D point sets including
point-to-point distances and angles.
"""
import itertools
from math import pi

import numpy as np

from pylocus.settings import *


class PointSet:
    """Class describing a typical point configuration.

    :param self.N: Number of points.
    :param self.d: Dimension or points (typically 2 or 3).
    :param self.points: Matrix of points (self.N x self.d).
    :param self.edm: Matrix (self.Nx self.N) of squared distances (Euclidean distance matrix).
    """

    def __init__(self, N, d):
        self.N = N
        self.d = d
        self.points = np.empty([self.N, self.d])
        self.edm = np.empty([self.N, self.N])


    def copy(self):
        new = PointSet(self.N, self.d)
        new.points = self.points.copy()
        new.edm = self.edm.copy()
        return new


    def init(self):
        self.create_edm()


    def add_noise(self, noise, indices=None):
        if indices is None:
            indices = range(self.N)
        self.points = return_noisy_points(noise, indices, self.points.copy())
        self.init()


    def set_points(self, mode='', points=None, range_=RANGE, size=1):
        """ Initialize points according to predefined modes.

        :param range_:[xmin, xmax, ymin, ymax], range of point sets
        """
        if mode == 'last':
            if points is None:
                print('Error: empty last point specification given.')
                return
            tol = 0.1
            [i, j, k] = points
            alpha = 2.0 * np.random.rand(1) + 0.5
            beta = 2.0 * np.random.rand(1) + 0.5
            if i >= 0 and j < 0 and k < 0:
                # at one corner of triangle
                assert i < 3
                other = np.delete(np.arange(3), i)
                u = (self.points[i, :] - self.points[other[0], :]
                     ) / np.linalg.norm(
                         self.points[i, :] - self.points[other[0], :])
                v = (self.points[i, :] - self.points[other[1], :]
                     ) / np.linalg.norm(
                         self.points[i, :] - self.points[other[1], :])
                self.points[-1, :] = self.points[i, :] + alpha * u + beta * v
            elif i >= 0 and j >= 0 and k < 0:
                found = False
                safety_it = 0
                while not found:
                    alpha = np.random.uniform(tol, 1 - tol)
                    beta = 1.0 - alpha
                    gamma = 2 * np.random.rand(1) + tol
                    assert j < 3
                    other = np.delete(np.arange(3), (i, j))
                    u = (
                        self.points[i, :] - self.points[other, :]
                    )  # /np.linalg.norm(self.points[i,:] - self.points[other,:])
                    v = (
                        self.points[j, :] - self.points[other, :]
                    )  # /np.linalg.norm(self.points[j,:] - self.points[other,:])
                    self.points[-1, :] = (1.0 + gamma) * (
                        self.points[other, :] + alpha * u + beta * v)
                    #check if new direction lies between u and v.
                    new_direction = self.points[-1, :] - self.points[other, :]
                    new_direction = new_direction.cp.reshape(
                        (-1, )) / np.linalg.norm(new_direction)
                    u = u.cp.reshape((-1, )) / np.linalg.norm(u)
                    v = v.cp.reshape((-1, )) / np.linalg.norm(v)
                    #print('{} + {} = {}'.format(acos(np.dot(new_direction,u)),acos(np.dot(new_direction,v)),acos(np.dot(u,v))))
                    if abs(
                            acos(np.dot(new_direction, u)) +
                            acos(np.dot(new_direction, v)) -
                            acos(np.dot(u, v))) < 1e-10:
                        found = True
                    safety_it += 1
                    if safety_it > 100:
                        print('Error: nothing found after 100 iterations.')
                        return
            elif i >= 0 and j >= 0 and k >= 0:
                # inside triangle
                assert k < 3
                found = False
                safety_it = 0
                while not found:
                    alpha = np.random.rand(1) + tol
                    beta = np.random.rand(1) + tol
                    other = np.delete(np.arange(3), i)
                    u = self.points[other[0], :] - self.points[i, :]
                    v = self.points[other[1], :] - self.points[i, :]
                    temptative_point = self.points[i, :] + alpha * u + beta * v
                    vjk = self.points[other[1], :] - self.points[other[0], :]
                    njk = [vjk[1], -vjk[0]]
                    if (np.dot(self.points[j, :] - self.points[i, :], njk) >
                            0) != (np.dot(temptative_point - self.points[j, :],
                                          njk) > 0):
                        self.points[-1, :] = temptative_point
                        found = True
                    safety_it += 1
                    if safety_it > 100:
                        print('Error: nothing found after 100 iterations.')
                        return
            elif i < 0 and j < 0 and k < 0:
                x = range_[0] + (
                    range_[1] - range_[0]) * np.random.rand(1)
                y = range_[2] + (
                    range_[1] - range_[0]) * np.random.rand(1)
                self.points[-1, :] = [x, y]
            else:
                print("Error: non-valid arguments.")
        elif mode == 'random':
            """ Create N uniformly distributed points in [0, size] x [0, size]
            """
            self.points = np.random.uniform(0, size, (self.N, self.d))
        elif mode == 'normal':
            self.points = np.random.normal(0, size, (self.N, self.d))
        elif mode == 'circle':
            from math import cos, sin
            x_range = size / 2.0
            y_range = size / 2.0
            c = np.array((x_range, y_range))
            r = 0.9 * min(x_range, y_range)
            theta = 2 * pi / self.N
            for i in range(self.N):
                theta_tot = i * theta
                self.points[i, :] = c + np.array(
                    (r * cos(theta_tot), r * sin(theta_tot)))
        elif mode == 'set':
            """
            Place points according to hard coded rule.
            """
            if self.N == 3:
                x = [-1.0, 1.0, 0.0]
                y = [-1.0, -1.0, 1.0]
            elif self.N == 4:
                x = [-1.0, 1.0, 0.0, 0.0]
                y = [-1.0, -1.0, 1.0, 0.0]
            elif self.N == 5:
                x = [-0.0, 1.5, 1.5, -0.0, -1.0]
                y = [-1.0, -1.0, 1.0, 1.0, 0.0]
            else:
                print("Error: No rule defined for N = ", self.N)
                return
            self.points = np.c_[x, y]
        elif mode == 'geogebra':
            if self.N == 4:
                self.points = np.array(((1.5, 1.8), (7.9, 2.5), (2.3, 5.1),
                                        (3.34, -1.36)))
            elif self.N == 5:
                self.points = np.array(((1.5, 1.8), (7.9, 2.5), (2.3, 5.1),
                                        (3.34, -1.36), (5, 1.4)))
            else:
                print("Error: No rule defined for N = ", self.N)
        elif mode == '':
            if points is None:
                raise NotImplementedError("Need to give either mode or points.")
            else:
                self.points = points
                self.N, self.d = points.shape

        self.init()


    def create_edm(self):
        from pylocus.basics import get_edm
        self.edm = get_edm(self.points)


    def get_abs_angles(self):
        from pylocus.basics_angles import get_absolute_angle
        abs_angles = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(i, self.N):
                abs_angles[i, j] = get_absolute_angle(self.points[i, :],
                                                      self.points[j, :])
                abs_angles[j, i] = get_absolute_angle(self.points[j, :],
                                                      self.points[i, :])
                if j > i:
                    assert abs(abs_angles[i, j] - abs_angles[j, i]) - pi < 1e-10, \
                        "Angles do not add up to pi: %r-%r=%r" % \
                        (abs_angles[i, j], abs_angles[j, i],
                         abs_angles[i, j] - abs_angles[j, i])
        return abs_angles


    def plot_all(self, title='', size=[5, 2], filename='', axis='off'):
        from .plots_cti import plot_points
        plot_points(self.points, title, size, filename, axis)

    def plot_some(self, range_, title='', size=[5, 2]):
        from .plots_cti import plot_points
        plot_points(self.points[range_, :], title, size)


class HeterogenousSet(PointSet):
    """ Class containing heteregenous information in the form of direction vectors. 

    :param self.m: Number of edges.
    :param self.V: Matrix of edges (self.m x self.d)
    :param self.KE: dissimilarity matrix (self.m x self.m) 
    :param self.C: matrix for getting edges from points (self.V = self.C.dot(self.X))
    :param self.dm: vector containing lengths of edges (self.m x 1)
    :param self.Om: Matrix containing cosines of inner angles (self.m x self.m)
    """

    def __init__(self, N, d):
        PointSet.__init__(self, N, d)
        self.m = int((self.N - 1) * self.N / 2.0)
        self.V = np.zeros((self.m, d))
        self.KE = np.zeros((self.m, self.m))
        self.C = np.zeros((self.m, self.N))
        self.dm = np.zeros((self.m, 1))
        self.Om = np.zeros((self.m, self.m))

    def init(self):
        PointSet.init(self)
        self.create_V()
        self.create_Om()

    def create_V(self):
        start = 0
        for i in range(self.N):
            n = self.N - i - 1
            self.C[start:start + n, i] = 1
            self.C[start:start + n, i + 1:] = -np.eye(n)
            start = start + n
        self.V = np.dot(self.C, self.points)
        self.KE = np.dot(self.V, self.V.T)
        self.dm = np.linalg.norm(self.V, axis=1)

    def create_Om(self):
        for i in range(self.m):
            for j in range(self.m):
                if i != j:
                    norm = np.linalg.norm(
                        self.V[i, :]) * np.linalg.norm(self.V[j, :])
                    cos_inner_angle = np.dot(self.V[i, :], self.V[
                                             j, :]) / norm
                else:
                    cos_inner_angle = 1.0
                self.Om[i, j] = cos_inner_angle

    def get_KE_constraints(self):
        """Get linear constraints on KE matrix.
        """
        C2 = np.eye(self.m)
        C2 = C2[:self.m - 2, :]
        to_be_deleted = []
        for idx_vij_1 in range(self.m - 2):
            idx_vij_2 = idx_vij_1 + 1
            C2[idx_vij_1, idx_vij_2] = -1
            i1 = np.where(self.C[idx_vij_1, :] == 1)[0][0]
            i2 = np.where(self.C[idx_vij_2, :] == 1)[0][0]
            j = np.where(self.C[idx_vij_1, :] == -1)[0][0]
            if i1 == i2:
                i = i1
                k = np.where(self.C[idx_vij_2, :] == -1)[0][0]
                i_indices = self.C[:, j] == 1
                j_indices = self.C[:, k] == -1
                idx_vij_3 = np.where(np.bitwise_and(
                    i_indices, j_indices))[0][0]
                #print('v{}{}, v{}{}, v{}{}\n{}    {}    {}'.format(j,i,k,i,k,j,idx_vij_1,idx_vij_2,idx_vij_3))
                C2[idx_vij_1, idx_vij_3] = 1
            else:
                #print('v{}{}, v{}{} not considered.'.format(j,i1,j,i2))
                to_be_deleted.append(idx_vij_1)
        C2 = np.delete(C2, to_be_deleted, axis=0)
        b = np.zeros((C2.shape[0], 1))
        return C2, b

    def copy(self):
        new = HeterogenousSet(self.N, self.d)
        new.points = self.points.copy()
        new.init()
        return new


def dm_from_edm(edm):
    from pylocus.basics import vector_from_matrix
    dm = vector_from_matrix(edm)
    dm = np.extract(dm > 0, dm)
    return np.power(dm, 0.5)


def edm_from_dm(dm, N):
    from pylocus.basics import matrix_from_vector
    edm_upper = matrix_from_vector(dm, N)
    edm = np.power(edm_upper + edm_upper.T, 2.0)
    return edm


def sdm_from_dmi(dmi, N):
    from pylocus.basics import matrix_from_vector
    sdm_upper = matrix_from_vector(dmi, N)
    sdm = sdm_upper - sdm_upper.T
    # assure diagonal is zero
    np.fill_diagonal(sdm, 0)
    return sdm


def get_V(anglesm, dm):
    V = np.c_[-np.multiply(np.cos(anglesm), dm),
              -np.multiply(np.sin(anglesm), dm)]
    return V


def dmi_from_V(V, dimension):
    dmi = V[:, dimension]
    return dmi


def create_from_points(points, PointClass):
    new = PointClass(points.shape[0], points.shape[1])
    new.points = points
    new.init()
    return new


def return_noisy_points(noise, indices, points):
    d = points.shape[1]
    points[indices, :] += np.random.normal(0, noise, (len(indices), d))
    return points
