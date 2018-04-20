#!/usr/bin/env python
# module LATERATION

import numpy as np
from scipy.linalg import eigvals, eigvalsh
from .basics import assert_print, assert_all_print
from cvxpy import *


def get_lateration_parameters(real_points, indices, index, edm, W=None):
    """ Get parameters relevant for lateration from full real_points, edm and W.
    """
    if W is None:
        W = np.ones(edm.shape)

    # delete points that are not considered anchors
    anchors = np.delete(real_points, indices, axis=0)
    r2 = np.delete(edm[index, :], indices)
    w = np.delete(W[index, :], indices)

    # set w to zero where measurements are invalid
    if np.isnan(r2).any():
        nan_measurements = np.where(np.isnan(r2))[0]
        r2[nan_measurements] = 0.0
        w[nan_measurements] = 0.0
    if np.isnan(w).any():
        nan_measurements = np.where(np.isnan(w))[0]
        r2[nan_measurements] = 0.0
        w[nan_measurements] = 0.0

    # delete anchors where weight is zero to avoid ill-conditioning
    missing_anchors = np.where(w == 0.0)[0]
    w = np.asarray(np.delete(w, missing_anchors))
    r2 = np.asarray(np.delete(r2, missing_anchors))
    w.resize(edm.shape[0] - len(indices) - len(missing_anchors), 1)
    r2.resize(edm.shape[0] - len(indices) - len(missing_anchors), 1)
    anchors = np.delete(anchors, missing_anchors, axis=0)
    assert w.shape[0] == anchors.shape[0]
    assert np.isnan(w).any() == False
    assert np.isnan(r2).any() == False
    return anchors, w, r2


def SRLS(anchors, w, r2, rescale=False, z=None, print_out=False):
    '''Squared range least squares (SRLS)

    Algorithm written by A.Beck, P.Stoica in "Approximate and Exact solutions of Source Localization Problems".

    :param anchors: anchor points (Nxd)
    :param w: weights for the measurements (Nx1)
    :param r2: squared distances from anchors to point x. (Nx1)
    :param rescale: Optional parameter. When set to True, the algorithm will
        also identify if there is a global scaling of the measurements.  Such a
        situation arise for example when the measurement units of the distance is
        unknown and different from that of the anchors locations (e.g. anchors are
        in meters, distance in centimeters).
    :param z: Optional parameter. Use to fix the z-coordinate of localized point.
    :param print_out: Optional parameter, prints extra information.

    :return: estimated position of point x.
    '''
    def y_hat(_lambda):
        lhs = ATA + _lambda * D
        assert A.shape[0] == b.shape[0]
        assert A.shape[1] == f.shape[0], 'A {}, f {}'.format(A.shape, f.shape)
        rhs = (np.dot(A.T, b) - _lambda * f).reshape((-1,))
        assert lhs.shape[0] == rhs.shape[0], 'lhs {}, rhs {}'.format(
            lhs.shape, rhs.shape)
        try:
            return np.linalg.solve(lhs, rhs)
        except:
            return np.zeros((lhs.shape[1],))

    def phi(_lambda):
        yhat = y_hat(_lambda).reshape((-1, 1))
        return np.dot(yhat.T, np.dot(D, yhat)) + 2 * np.dot(f.T, yhat)

    def phi_prime(_lambda):
        # TODO: test this.
        B = np.linalg.inv(ATA + _lambda * D)
        C = A.T.dot(b) - _lambda*f
        y_prime = -B.dot(D.dot(B.dot(C)) - f)
        y = y_hat(_lambda)
        return 2*y.T.dot(D).dot(y_prime) + 2*f.T.dot(y_prime)

    from scipy import optimize
    from scipy.linalg import sqrtm

    n, d = anchors.shape
    assert r2.shape[1] == 1 and r2.shape[0] == n, 'r2 has to be of shape Nx1'
    assert w.shape[1] == 1 and w.shape[0] == n, 'w has to be of shape Nx1'
    if z is not None:
        assert d == 3, 'Dimension of problem has to be 3 for fixing z.'

    if rescale and z is not None:
        raise NotImplementedError('Canoot run rescale for fixed z.')
    if rescale and n < d + 2:
        raise ValueError(
            'A minimum of d + 2 ranges are necessary for rescaled ranging.')
    elif n < d + 1:
        raise ValueError(
            'A minimum of d + 1 ranges are necessary for ranging.')

    Sigma = np.diagflat(np.power(w, 0.5))

    if rescale:
        A = np.c_[-2 * anchors, np.ones((n, 1)), -r2]
    else:
        if z is None:
            A = np.c_[-2 * anchors, np.ones((n, 1))]
        else:
            A = np.c_[-2 * anchors[:, :2], np.ones((n, 1))]

        A = Sigma.dot(A)

    if rescale:
        b = - np.power(np.linalg.norm(anchors, axis=1), 2).reshape(r2.shape)
    else:
        b = r2 - np.power(np.linalg.norm(anchors, axis=1), 2).reshape(r2.shape)
        if z is not None:
            b = b + 2 * anchors[:, 2].reshape((-1, 1)) * z - z**2
        b = Sigma.dot(b)

    ATA = np.dot(A.T, A)

    if rescale:
        D = np.zeros((d + 2, d + 2))
        D[:d, :d] = np.eye(d)
    else:
        if z is None:
            D = np.zeros((d + 1, d + 1))
        else:
            D = np.zeros((d, d))
        D[:-1, :-1] = np.eye(D.shape[0]-1)

    if rescale:
        f = np.c_[np.zeros((1, d)), -0.5, 0.].T
    elif z is None:
        f = np.c_[np.zeros((1, d)), -0.5].T
    else:
        f = np.c_[np.zeros((1, 2)), -0.5].T

    eig = np.sort(np.real(eigvalsh(a=D, b=ATA)))
    if (print_out):
        print('ATA:', ATA)
        print('rank:', np.linalg.matrix_rank(A))
        print('eigvals:', eigvals(ATA))
        print('condition number:', np.linalg.cond(ATA))
        print('generalized eigenvalues:', eig)
    eps = 0.01
    if eig[-1] > 1e-10:
        I_orig = -1.0 / eig[-1] + eps
    else:
        print('Warning: biggest eigenvalue is zero!')
        I_orig = -1e-5
    inf = 1e5
    xtol = 1e-12
    try:
        lambda_opt = optimize.bisect(phi, I_orig, inf, xtol=xtol)
    except:
        print('Bisect failed. Trying Newton...')
        lambda_opt = I_orig
        try:
            lambda_opt = optimize.newton(
                phi, I_orig, fprime=phi_prime, maxiter=1000, tol=xtol)
            assert phi(lambda_opt) < xtol, 'did not find solution of phi(lambda)=0:={}'.format(
                phi(lambda_opt))
        except:
            print('SRLS ERROR: Did not converge. Setting lambda to 0.')
            lambda_opt = 0

    if (print_out):
        print('phi(I_orig)', phi(I_orig))
        print('phi(inf)', phi(inf))
        print('phi(opt)', phi(lambda_opt))

    # Compute best estimate
    yhat = y_hat(lambda_opt)

    if print_out and rescale:
        print('Scaling factor :', yhat[-1])

    if rescale:
        return yhat[:d], yhat[-1]
    elif z is None:
        return yhat[:d]
    else:
        return np.r_[yhat[0], yhat[1], z]


def PozyxLS(anchors, W, r2, print_out=False):
    ''' Algorithm used by pozyx (https://www.pozyx.io/Documentation/how_does_positioning_work)

    :param anchors: anchor points
    :param r2: squared distances from anchors to point x.

    :returns: estimated position of point x.
    '''
    N = anchors.shape[0]
    anchors_term = np.sum(np.power(anchors[:-1], 2), axis=1)
    last_term = np.sum(np.power(anchors[-1], 2))
    b = r2[:-1] - anchors_term + last_term - r2[-1]
    A = -2 * (anchors[:-1] - anchors[-1])
    x, res, rank, s = np.linalg.lstsq(A, b)
    return x


def RLS(anchors, W, r, print_out=False, grid=None, num_points=10):
    """ Range least squares (RLS) using grid search.

    Algorithm written by A.Beck, P.Stoica in "Approximate and Exact solutions of Source Localization Problems".

    :param anchors: anchor points
    :param r2: squared distances from anchors to point x.
    :param grid: where to search for solution.  (min, max) where min and max  are 
    lists of d elements, d being the dimension of the setup. If not given, the 
    search is conducted within the space covered by the anchors.
    :param num_points: number of grid points per direction.

    :return: estimated position of point x.
    """
    def cost_function(arr):
        X = np.c_[arr]
        r_measured = np.linalg.norm(anchors - X)
        mse = np.linalg.norm(r_measured - r)**2
        return mse

    if grid is None:
        grid = [np.min(anchors, axis=0), np.max(anchors, axis=0)]

    d = anchors.shape[1]
    x = np.linspace(grid[0][0], grid[1][0], num_points)
    y = np.linspace(grid[0][1], grid[1][1], num_points)
    if d == 2:
        errors_test = np.zeros((num_points, num_points))
        for i, xs in enumerate(x):
            for j, ys in enumerate(y):
                errors_test[i, j] = cost_function((xs, ys))
        min_idx = errors_test.argmin()
        min_idx_multi = np.unravel_index(min_idx, errors_test.shape)
        xhat = np.c_[x[min_idx_multi[0]], y[min_idx_multi[1]]]
    elif d == 3:
        z = np.linspace(grid[0][2], grid[1][2], num_points)
        errors_test = np.zeros((num_points, num_points, num_points))

        # TODO: make this more efficient.
        #xx, yy, zz= np.meshgrid(x, y, z)
        for i, xs in enumerate(x):
            for j, ys in enumerate(y):
                for k, zs in enumerate(z):
                    errors_test[i, j, k] = cost_function((xs, ys, zs))
        min_idx = errors_test.argmin()
        min_idx_multi = np.unravel_index(min_idx, errors_test.shape)
        xhat = np.c_[x[min_idx_multi[0]],
                     y[min_idx_multi[1]], z[min_idx_multi[2]]]
    else:
        raise ValueError('Non-supported number of dimensions.')
    return xhat[0]


def RLS_SDR(anchors, W, r, print_out=False):
    """ Range least squares (RLS) using SDR.

    Algorithm cited by A.Beck, P.Stoica in "Approximate and Exact solutions of Source Localization Problems".

    :param anchors: anchor points
    :param r2: squared distances from anchors to point x.

    :return: estimated position of point x.
    """
    from pylocus.basics import low_rank_approximation, eigendecomp
    from pylocus.mds import x_from_eigendecomp

    m = anchors.shape[0]
    d = anchors.shape[1]

    G = Variable(m + 1, m + 1)
    X = Variable(d + 1, d + 1)
    constraints = [G[m, m] == 1.0,
                   X[d, d] == 1.0,
                   G >> 0, X >> 0,
                   G == G.T, X == X.T]
    for i in range(m):
        Ci = np.eye(d + 1)
        Ci[:-1, -1] = -anchors[i]
        Ci[-1, :-1] = -anchors[i].T
        Ci[-1, -1] = np.linalg.norm(anchors[i])**2
        constraints.append(G[i, i] == trace(Ci * X))

    obj = Minimize(trace(G) - 2 * sum_entries(mul_elemwise(r, G[m, :-1].T)))
    prob = Problem(obj, constraints)

    ## Solution
    total = prob.solve(verbose=True)
    rank_G = np.linalg.matrix_rank(G.value)
    rank_X = np.linalg.matrix_rank(X.value)
    if rank_G > 1:
        u, s, v = np.linalg.svd(G.value, full_matrices=False)
        print('optimal G is not of rank 1!')
        print(s)
    if rank_X > 1:
        u, s, v = np.linalg.svd(X.value, full_matrices=False)
        print('optimal X is not of rank 1!')
        print(s)

    factor, u = eigendecomp(X.value, 1)
    xhat = x_from_eigendecomp(factor, u, 1)
    return xhat
