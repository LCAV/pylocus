#!/usr/bin/env python
# TODO: add below to templates. Add docstrings to code snippets module.
""" This is an example module
"""
__version__ = '0.1'
__author__ = 'Frederike Duembgen'

import numpy as np
import settings
from math import pi

class PointConfiguration:
    """Class describing a typical point configuration.

    TODO: add a long description here.

    Attributes:
        self.N: Number of points.
        self.d: Dimension or points (typically 2 or 3).
        self.T: Number of triangles.
        self.M: Number of inner angles.
        self.points: Matrix of points (self.N x self.d).
        self.theta: Vector of inner angles.
        self.corners: Matrix of corners corresponding to inner angles. Row (k,i,j) corresponds to theta_k(i,j).
        self.abs_angles: Matrix (self.N x self.N) of absolute angles. Element (i,j) corresponds to absolute angle from origin to ray from point i to point j.
        self.edm: Matrix (self.Nx self.N) of squared distances (Euclidean distance matrix).
    """

    def __init__(self, N, d):
        from scipy.special import binom
        self.N = N
        self.d = d
        self.T = int(binom(self.N, 3))
        self.M = int(3 * self.T)
        self.points = np.empty([self.N, self.d])
        self.theta = np.empty([
            self.M,
        ])
        self.corners = np.empty([self.M, 3])
        self.abs_angles = np.empty([self.N, self.N])
        self.edm = np.empty([self.N, self.N])

    def copy(self):
        new = PointConfiguration(self.N, self.d)
        new.points = self.points.copy()
        new.theta = self.theta.copy()
        new.corners = self.corners.copy()
        new.abs_angles = self.abs_angles.copy()
        new.edm = self.edm.copy()
        return new

    def init(self):
        self.create_edm()
        self.create_abs_angles()
        self.create_theta()

    def add_noise(self, noise, mode='points'):
        if mode == 'points':
            self.points += np.random.normal(0, noise, (self.N, self.d))

    def return_noisy(self, noise, mode='noisy', idx=0, visualize=False):
        if mode == 'normal':
            theta = self.theta.copy() + np.random.normal(0, noise, self.M)
            if (visualize):
                plot_thetas([self_theta, theta], ['original', 'noise'])
            return theta
        if mode == 'constant':
            theta = self.theta.copy() + noise
            if (visualize):
                plot_thetas([self_theta, theta], ['original', 'noise'])
            return theta
        if mode == 'punctual':
            theta = self.theta.copy()
            theta[idx] += noise
            if (visualize):
                plot_thetas_in_one([self.theta, theta], ['original', 'noise'])
            return theta

    def set_points(self, mode, points=None):
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
                    # TODO: this doesn't end up on the good side of the triangle.
                    # on one side of triangle
                    alpha = np.random.uniform(tol, 1 - tol)
                    beta = 1.0 - alpha
                    gamma = 2 * np.random.rand(1) + tol
                    assert j < 3
                    other = np.delete(np.arange(3), (i, j))
                    u = (
                        self.points[i, :] - self.points[other, :]
                    )  #/np.linalg.norm(self.points[i,:] - self.points[other,:])
                    v = (
                        self.points[j, :] - self.points[other, :]
                    )  #/np.linalg.norm(self.points[j,:] - self.points[other,:])
                    self.points[-1, :] = (1.0 + gamma) * (
                        self.points[other, :] + alpha * u + beta * v)
                    #check if new direction lies between u and v.
                    new_direction = self.points[-1, :] - self.points[other, :]
                    new_direction = new_direction.reshape(
                        (-1, )) / np.linalg.norm(new_direction)
                    u = u.reshape((-1, )) / np.linalg.norm(u)
                    v = v.reshape((-1, )) / np.linalg.norm(v)
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
                x = settings.RANGE[0] + (
                    settings.RANGE[1] - settings.RANGE[0]) * np.random.rand(1)
                y = settings.RANGE[2] + (
                    settings.RANGE[1] - settings.RANGE[0]) * np.random.rand(1)
                self.points[-1, :] = [x, y]
            else:
                print("Error: non-valid arguments.")
        elif mode == 'random':
            """
            Create N uniformly distributed points in
            [xlim[0],ylim[0]] x [xlim[1],ylim[1]]
            """
            x = settings.RANGE[0] + (settings.RANGE[1] - settings.RANGE[0]
                                     ) * np.random.rand(self.N)
            y = settings.RANGE[2] + (settings.RANGE[1] - settings.RANGE[0]
                                     ) * np.random.rand(self.N)
            self.points = np.c_[x, y]
        elif mode == 'normal':
            self.points = np.random.normal(0, 1.0, (self.N, self.d))
        elif mode == 'circle':
            x_range = (settings.RANGE[1] - settings.RANGE[0]) / 2.0
            y_range = (settings.RANGE[3] - settings.RANGE[2]) / 2.0
            c = np.array((settings.RANGE[0] + x_range,
                          settings.RANGE[2] + y_range))
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
        self.init()
    
# TODO: Which of the two below should be used? 

    def get_inner_angle(self, corner, other):
        from basics import get_inner_angle
        return get_inner_angle(self.points[corner, :], (
            self.points[other[0], :], self.points[other[1], :]))

    def get_theta(self, i, j, k):
        combination = np.array([i, j, k])
        idx = np.all(self.corners == combination, axis=1)
        return self.theta[idx][0]

    def get_orientation(k, i, j):
        from basics import from_0_to_2pi
        """calculate angles theta_ik and theta_jk theta produce point Pk.
        Should give the same as get_absolute_angle! """
        theta_ij = own.abs_angles[i, j]
        theta_ji = own.abs_angles[j, i]

        # complicated
        xi = own.points[i, 0]
        xj = own.points[j, 0]
        yi = own.points[i, 1]
        yj = own.points[j, 1]
        w = np.array([yi - yj, xj - xi])
        test = np.dot(own.points[k, :] - own.points[i, :], w) > 0

        # more elegant
        theta_ik = truth.abs_angles[i, k]
        diff = from_0_to_2pi(theta_ik - theta_ij)
        test2 = (diff > 0 and diff < pi)
        assert (test == test2), "diff: %r, scalar prodcut: %r" % (diff, np.dot(
            own.points[k, :] - own.points[i, :], w))

        thetai_jk = truth.get_theta(i, j, k)
        thetaj_ik = truth.get_theta(j, i, k)
        if test:
            theta_ik = theta_ij + thetai_jk
            theta_jk = theta_ji - thetaj_ik
        else:
            theta_ik = theta_ij - thetai_jk
            theta_jk = theta_ji + thetaj_ik
        theta_ik = from_0_to_2pi(theta_ik)
        theta_jk = from_0_to_2pi(theta_jk)
        return theta_ik, theta_jk


    def create_edm(self):
        rows, cols = np.indices((self.N, self.N))
        self.edm = np.sum(
            (self.points[rows, :] - self.points[cols, :])**2, axis=2)

    def create_abs_angles(self):
        from basics import get_absolute_angle
        abs_angles = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(i, self.N):
                abs_angles[i, j] = get_absolute_angle(self.points[i, :],
                                                      self.points[j, :])
                abs_angles[j, i] = get_absolute_angle(self.points[j, :],
                                                      self.points[i, :])
                if j > i:
                    assert abs(abs_angles[i,j] - abs_angles[j,i]) - pi < 1e-10, \
                        "Angles do not add up to pi: %r-%r=%r" % \
                        (abs_angles[i,j], abs_angles[j,i], \
                        abs_angles[i,j] - abs_angles[j,i])
        self.abs_angles = abs_angles

    def create_abs_angles_from_edm(self):
        """TODO: which one is better?"""
        rows, cols = np.indices((self.N, self.N))
        pi_pj_x = (self.points[rows, 0] - self.points[cols, 0])
        pi_pj_y = (self.points[rows, 1] - self.points[cols, 1])
        D = np.sqrt(
            np.sum((self.points[rows, :] - self.points[cols, :])**2, axis=2))
        cosine = np.ones([self.N, self.N])
        sine = np.zeros([self.N, self.N])
        cosine[D > 0] = pi_pj_x[D > 0] / D[D > 0]
        sine[D > 0] = pi_pj_y[D > 0] / D[D > 0]
        Dc = acos(cosine)
        for i in range(Dc.shape[0]):
            for j in range(Dc.shape[0]):
                if cosine[i, j] < 0 and sine[i, j] < 0:
                    # angle between pi and 3pi/2
                    Dc[i, j] = 2 * pi - Dc[i, j]
                if cosine[i, j] > 0 and sine[i, j] < 0:
                    # angle between 3pi/2 and 2pi
                    Dc[i, j] = 2 * pi - Dc[i, j]
        self.abs_angles = Dc

    def create_theta(self):
        """
        Returns the set of inner angles (between 0 and pi)
        reconstructed from point coordinates.
        Also returns the corners corresponding to each entry of theta.
        """
        import itertools
        from basics import from_0_to_pi
        theta = np.empty((self.M, ))
        corners = np.empty((self.M, 3))
        k = 0
        indices = np.arange(self.N)
        for triangle in itertools.combinations(indices, 3):
            for counter, idx in enumerate(triangle):
                corner = idx
                other = np.delete(triangle, counter)
                corners[k, :] = [corner, other[0], other[1]]
                theta[k] = self.get_inner_angle(corner, other)
                theta[k] = from_0_to_pi(theta[k])
                if settings.DEBUG:
                    print(self.abs_angles[corner, other[0]], \
                            self.abs_angles[corner, other[1]])
                    print('theta', corners[k, :], theta[k])
                k = k + 1
            inner_angle_sum = theta[k - 1] + theta[k - 2] + theta[k - 3]
            assert abs(inner_angle_sum - pi) < 1e-10, \
                    'inner angle sum: {} {} {}'.format(triangle, inner_angle_sum,(theta[k-1],theta[k-2],theta[k-3]))
        self.theta = theta
        self.corners = corners
        return theta, corners

# TODO: where do I need the three below?

    def get_tensor_edm(self):
        D = np.empty([self.N * self.d, self.N * self.d])
        for i in range(self.N):
            for j in range(self.N):
                starti = i * self.d
                endi = (i + 1) * (self.d)
                startj = j * self.d
                endj = (j + 1) * self.d
                a = np.outer(self.points[i, :], self.points[j, :])
                b = np.outer(self.points[j, :], self.points[j, :])
                c = np.outer(self.points[i, :], self.points[i, :])
                D[starti:endi, startj:endj] = b + c - 2 * a
        return D

    def get_closed_form(self, edm):
        """ TODO: what is this? """
        Daug = self.get_tensor_edm()
        T = np.empty([self.N, self.N, self.N])
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    if j == i or k == i:
                        factor = 0.0
                    else:
                        factor = 1.0 / (self.edm[j, i] * self.edm[k, i])
                    diff = self.points[k, :] - self.points[i, :]
                    starti = i * self.d
                    endi = (i + 1) * (self.d)
                    startj = j * self.d
                    endj = (j + 1) * self.d
                    Dij = Daug[starti:endi, startj:endj]
                    T[i, j, k] = factor * np.dot(diff.T, np.dot(Dij, diff))
        return T

    def get_theta_tensor(self):
        self.theta_tensor = get_theta_tensor(self.theta, self.corners, self.N)
        return self.theta_tensor

# TODO: This is for iterative algorithm only...

    def get_indices(self, k):
        """ Get indices of theta vector that have k as first corner.
        Args:
            k: Index of corner.
        Returns:
            indices_rays: Indices of ray angles in theta vector.
            indices_triangles: Indices of triangle angles in theta vector.
            corners_rays: List of corners of ray angles.
            angles_rays: List of corners of triangles angles.
        """
        indices_rays = []
        indices_triangles = []
        corners_rays = []
        angles_rays = []
        for t, triangle in enumerate(self.corners):
            if triangle[0] == k:
                indices_rays.append(t)
                corners_rays.append(triangle)
                angles_rays.append(self.theta[t])
            else:
                indices_triangles.append(t)
        np_corners_rays = np.vstack(corners_rays)
        np_angles_rays = np.vstack(angles_rays).reshape((-1, ))
        return indices_rays, indices_triangles, np_corners_rays, np_angles_rays

    def get_G(self, k, add_noise=True):
        """ get G matrix from angles. """
        G = np.ones((self.N - 1, self.N - 1))
        if (add_noise):
            noise = pi * 0.1 * np.random.rand(
                (self.N - 1) * (self.N - 1)).reshape((self.N - 1, self.N - 1))
        other_indices = np.delete(range(self.N), k)
        for idx, i in enumerate(other_indices):
            for jdx, j in enumerate(other_indices):
                if (add_noise and
                        i != j):  # do not add noise on diagonal elements.
                    thetak_ij = self.get_inner_angle(k,
                                                     (i, j)) + noise[idx, jdx]
                else:
                    thetak_ij = self.get_inner_angle(k, (i, j))
                G[idx, jdx] = cos(thetak_ij)
                G[jdx, idx] = cos(thetak_ij)
        return G

# TODO: Which of these two is better? And should they really be in this class? 

    def reconstruct_from_inner_angles(self, theta):
        from algorithms import reconstruct_from_inner_angles
        from algorithms import procrustes
        theta_tensor = get_theta_tensor(theta, self.corners, self.N)
        reconstruction = reconstruct_from_inner_angles(
            self.points[0, :], self.points[1, :], self.abs_angles[0, 2],
            self.abs_angles[1, 2], theta_tensor)
        new_points, __, __, __  = procrustes(self.points,reconstruction.points,scale=True)
        reconstruction.points=new_points
        reconstruction.init()
        return reconstruction

    def reconstruct(self, theta):
        from algorithms import reconstruct
        i = 0
        j = 1
        theta_tensor = get_theta_tensor(theta, self.corners, self.N)
        Pi = self.points[i, :]
        Pj = self.points[j, :]
        k = 2
        Pk = self.points[k, :]
        reconstruction = reconstruct(Pi, Pj, i, j, theta_tensor, Pk, k)
        return reconstruction

    def plot_all(self, title='', size=[5, 2], filename=''):
        from plots import plot_points
        plot_points(self.points, title, size, filename)

    def plot_some(self, range_, title='', size=[5, 2]):
        from plots import plot_points
        plot_points(self.points[range_, :], title, size)

class ConstrainedConfiguration(PointConfiguration):
    """ Class with linear constraints."

    TODO: Add longer description.

    Attributes:
        self.C: Number of linear constraints.
        self.A: Matrix of constraints (self.C x self.M)
        self.b: Vector of constraints (self.C x 1)
    """
    def __init__(self, N, d):
        PointConfiguration.__init__(self, N, d)
        self.C = 1
        self.A = np.empty((self.C, self.M))
        self.b = np.empty((self.C, 1))

    def get_convex_polygons(self, m, print_out=False):
        """
        Args:
            m: size of polygones (number of corners)
        Returns:
            convex_polygons: (ordered) indices of all convex polygones of size m.
        """
        convex_polygons = []
        for corners in itertools.combinations(np.arange(self.N), m):
            p = np.zeros(m, np.uint)
            p[0] = corners[0]
            left = corners[1:]
            # loop through second corners
            for i, second in enumerate(corners[1:m - 1]):
                p[1] = second
                left = np.delete(corners, (0, i + 1))
                for j, last in enumerate(corners[i + 2:]):
                    left = np.delete(corners, (0, i + 1, j + i + 2))
                    p[-1] = last
                    # loop through all permutations of left corners.
                    for permut in itertools.permutations(left):
                        p[2:-1] = permut
                        sum_theta = 0
                        # sum over all inner angles.
                        for k in range(m):
                            sum_theta += self.get_inner_angle(
                                p[1], (p[0], p[2]))
                            p = np.roll(p, 1)
                        angle = sum_theta
                        sum_angle = (m - 2) * pi
                        if (abs(angle - sum_angle) < 1e-14 or
                                abs(angle) < 1e-14):
                            if (print_out):
                                print("convex polygon found:    ", p)
                            convex_polygons.append(p.copy())
                        #  elif (angle < sum_angle):
                        #  if (print_out): print("non convex polygon found:",p,angle)
                        elif (angle > sum_angle):
                            if (print_out): print("oops")
        return convex_polygons

    def get_polygon_constraints(self,
                                range_polygones=range(3, 5),
                                print_out=False):
        """
        Args:
            range_polygones: list of numbers of polygones to test.
        Returns:
            A, b: the constraints on the theta-vector of the form A*theta = b
        """
        rows_A = []
        rows_b = []
        for m in range_polygones:
            if (print_out): print('checking {}-polygones'.format(m))
            polygons = self.get_convex_polygons(m)
            row_A, row_b = self.get_polygon_constraints_m(polygons, print_out)
            rows_A.append(row_A)
            rows_b.append(row_b)
        self.A = np.vstack(rows_A)
        self.b = np.hstack(rows_b)
        return self.A, self.b

    def get_angle_constraints_m(self, polygons_m, print_out=False):
        rows = []
        m = len(polygons_m[0])
        # initialization to empty led to A being filled with first row of currently stored A!
        A = np.zeros((1, self.M))
        b = np.empty((1, ))
        for p in polygons_m:
            if len(p) < 4:
                break
            if (print_out): print('sum of angles for p {}'.format(p))
            for j in p:
                if (print_out): print('for corner {}'.format(p[0]))
                k = m - 2  #for k in range(2, m-1): # how many angles to sum up.
                row = np.zeros(self.M)
                # outer angle
                for i in range(1, m - k):
                    sum_angles = 0
                    # inner angles
                    for l in range(i, i + k):
                        sum_angles += self.get_inner_angle(
                            p[0], (p[l], p[l + 1]))
                        index = get_index(self.corners, p[0], (p[l], p[l + 1]))
                        if (print_out):
                            print('+ {} (= index{}: {})'.format(
                                (p[0], (p[l], p[l + 1])),
                                np.where(index), self.corners[index, :]))
                        row[index] = 1
                    index = get_index(self.corners, p[0], (p[i], p[i + k]))
                    if (print_out):
                        print(' = {} (= index{}: {})'.format((p[0], (p[i], p[
                            i + k])), np.where(index), self.corners[index, :]))
                    row[index] = -1
                    rows.append(row)
                    if (print_out):
                        print('sum_angles - expected:{}'.format(
                            sum_angles - self.get_inner_angle(
                                p[0], (p[i], p[i + k]))))
                    if np.sum(np.nonzero(A)) == 0:
                        A = row
                    else:
                        A = np.vstack((A, row))
                p = np.roll(p, 1)
        if A.shape[0] > 0:
            b = np.zeros(A.shape[0])
        self.A = A
        self.b = b
        return A, b

    def get_polygon_constraints_m(self, polygons_m, print_out=False):
        """
        Args:
            range_polygones: list of numbers of polygones to test.
        Returns:
            A, b: the constraints on the theta-vector of the form A*theta = b
        """
        rows_b = []
        rows_A = []
        m = len(polygons_m[0])
        rows_b.append((m - 2) * pi * np.ones(
            len(polygons_m), ))
        for p in polygons_m:
            row = np.zeros((self.theta.shape[0], ))
            for k in range(m):
                index = get_index(self.corners, p[1], (p[0], p[2]))
                row[index] = 1
                p = np.roll(p, 1)
            assert np.sum(row) == m
            rows_A.append(row)
        A = np.vstack(rows_A)
        b = np.hstack(rows_b)
        num_constraints = A.shape[0]
        A_repeat = np.repeat(A.astype(bool), 3).reshape((1, -1))
        corners = self.corners.reshape((1, -1))
        corners_tiled = np.tile(corners, num_constraints)
        if (print_out): print('shape of A {}'.format(A.shape))
        if (print_out):
            print('chosen angles m={}:\n{}'.format(m, (corners_tiled)[A_repeat]
                                                   .reshape((-1, m * 3))))
        if (print_out): print('{}-polygones: {}'.format(m, rows_A))
        self.A = A
        self.b = b
        return A, b

class HeterogenousConfiguration(PointConfiguration):
    def __init__(self, N, d):
        PointConfiguration.__init__(self, N, d)
        #TODO this is wrong! Is it really? 
        self.m = int((self.N-1)*self.N/2.0)
        self.V = np.zeros((self.m, d))
        self.KE = np.zeros((self.m, self.m))
        self.C = np.zeros((self.m, self.N))
        self.dm = np.zeros((self.m, 1))
        self.Om = np.zeros((self.m, self.m))

    def init(self):
        PointConfiguration.init(self)
        start = 0
        for i in range(self.N):
            n = self.N - i - 1
            self.C[start:start+n,i] = 1
            self.C[start:start+n,i+1:] = -np.eye(n) 
            start = start + n
        self.V = np.dot(self.C, self.points)
        self.KE = np.dot(self.V, self.V.T)
        self.dm = np.linalg.norm(self.V, axis=1)
        for i in range(self.m):
            for j in range(self.m):
                if i!=j:
                    cos_inner_angle = np.dot(self.V[i,:],self.V[j,:]) / (np.linalg.norm(self.V[i,:])*np.linalg.norm(self.V[j,:]))
                else:
                    cos_inner_angle = 1.0
                self.Om[i,j] = cos_inner_angle

def dm_from_edm(edm):
    dm = np.triu(edm)
    dm = np.extract(dm>0, dm)
    return np.power(dm, 0.5)

if __name__ == "__main__":
    print('running module point_configuration.py')
