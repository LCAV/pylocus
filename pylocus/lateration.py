#!/usr/bin/env python
# module LATERATION

import numpy as np
from .basics import assert_print, assert_all_print
from cvxpy import *



def get_lateration_parameters(real_points, indices, index, edm, W=None):
    """ Get parameters relevant for lateration from full real_points, edm and W.
    """
    # delete points that are not considered anchors
    anchors = np.delete(real_points, indices, axis=0)
    r2 = np.delete(edm[index, :], indices)
    if W is None:
        W = np.ones(edm.shape)
    w = np.delete(W[index, :], indices)

    # delete anchors where weight is zero to avoid ill-conditioning
    missing_anchors = np.where(w == 0.0)
    w = np.delete(w, missing_anchors)
    r2 = np.delete(r2, missing_anchors)
    anchors = np.delete(anchors, missing_anchors, axis=0)
    return anchors, w, r2


def SRLS(anchors, W, r2, print_out=False):
    '''Squared range least squares (SRLS)

    Algorithm written by A.Beck, P.Stoica in "Approximate and Exact solutions of Source Localization Problems".

    :param anchors: anchor points
    :param r2: squared distances from anchors to point x.

    :return: estimated position of point x.
    '''
    def y_hat(_lambda):
        lhs = ATA + _lambda * D
        rhs = (np.dot(A.T, b).reshape((-1, 1)) - _lambda * f).reshape((-1,))
        return np.linalg.solve(lhs, rhs)

    def phi(_lambda):
        yhat = y_hat(_lambda).reshape((-1, 1))
        return np.dot(yhat.T, np.dot(D, yhat)) + 2 * np.dot(f.T, yhat)

    from scipy import optimize
    from scipy.linalg import sqrtm
    # Set up optimization problem
    n = anchors.shape[0]
    d = anchors.shape[1]
    A = np.c_[-2 * anchors, np.ones((n, 1))]
    Sigma = np.diag(np.power(W, 0.5))
    A = Sigma.dot(A)
    ATA = np.dot(A.T, A)
    b = r2 - np.power(np.linalg.norm(anchors, axis=1), 2)
    b = Sigma.dot(b)
    D = np.zeros((d + 1, d + 1))
    D[:d, :d] = np.eye(d)
    if (print_out):
        print('rank A:', A)
        print('ATA:', ATA)
        print('rank:', np.linalg.matrix_rank(A))
        print('ATA:', np.linalg.eigvals(ATA))
        print('D:', D)
        print('condition number:', np.linalg.cond(ATA))
    f = np.c_[np.zeros((1, d)), -0.5].T

    # Compute lower limit for lambda (s.t. AT*A+lambda*D psd)
    reg = 1
    sqrtm_ATA = sqrtm(ATA + reg * np.eye(ATA.shape[0]))
    B12 = np.linalg.inv(sqrtm_ATA)
    tmp = np.dot(B12, np.dot(D, B12))
    eig = np.linalg.eigvals(tmp)

    eps = 0.01
    if eig[0] == reg:
        print('eigenvalue equals reg.')
        I_orig = -1e5
    elif eig[0] > 1e-10:
        I_orig = -1.0 / eig[0] + eps
    else:
        print('smallest eigenvalue is zero')
        I_orig = -1e5
    inf = 1e5
    try:
        lambda_opt = optimize.bisect(phi, I_orig, inf)
    except:
        print('Bisect failed. Trying Newton...')
        lambda_opt = optimize.newton(phi, I_orig, maxiter=1000)
    if (print_out):
        print('I', I)
        print('phi I', phi(I))
        print('phi inf', phi(inf))
        print('lambda opt', lambda_opt)
        print('phi opt', phi(lambda_opt))
        xvec = np.linspace(I, lambda_opt + abs(I), 100)
        y = list(map(lambda x: phi(x), xvec))
        y = np.array(y).reshape((-1,))
        plt.plot(xvec, y)
        plt.grid('on')
        plt.vlines(lambda_opt, y[0], y[-1])
        plt.hlines(0, xvec[0], xvec[-1])

    # Compute best estimate
    yhat = y_hat(lambda_opt)

    # By definition, y = [x^T, alpha] and by constraints, alpha=norm(x)^2
    # TODO: why do these not work with weights?
    #assert_print(np.linalg.norm(yhat[:-1])**2 - yhat[-1], 1e-6)
    #assert_print(phi(lambda_opt), 1e-6)

    lhs = np.dot(A.T, A) + lambda_opt * D
    rhs = (np.dot(A.T, b).reshape((-1, 1)) - lambda_opt * f).reshape((-1,))
    assert_all_print(np.dot(lhs, yhat) - rhs, 1e-6)
    eig = np.array(np.linalg.eigvals(ATA + lambda_opt * D))
    #assert (eig >= 0).all(), 'not all eigenvalues positive:{}'.format(eig)
    return yhat[:d]


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
        for i,xs in enumerate(x):
            for j,ys in enumerate(y):
                errors_test[i,j] = cost_function((xs, ys))
        min_idx = errors_test.argmin()
        min_idx_multi = np.unravel_index(min_idx, errors_test.shape)
        xhat = np.c_[x[min_idx_multi[0]],y[min_idx_multi[1]]]
    elif d == 3:
        z = np.linspace(grid[0][2], grid[1][2], num_points)
        errors_test = np.zeros((num_points, num_points, num_points))

        # TODO: make this more efficient.
        #xx, yy, zz= np.meshgrid(x, y, z)
        for i,xs in enumerate(x):
            for j,ys in enumerate(y):
                for k,zs in enumerate(z):
                    errors_test[i,j,k] = cost_function((xs, ys, zs))
        min_idx = errors_test.argmin()
        min_idx_multi = np.unravel_index(min_idx, errors_test.shape)
        xhat = np.c_[x[min_idx_multi[0]],y[min_idx_multi[1]],z[min_idx_multi[2]]]
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

    m = anchors.shape[0]
    d = anchors.shape[1]

    G = Semidef(m + 1)
    X = Semidef(d + 1)

    constraints = [G[m,m] == 1,
                   X[d,d] == 1]
    for i in range(m):
        Ci = np.eye(d+1)
        Ci[:-1,-1] = -anchors[i]
        Ci[-1,:-1] = -anchors[i].T
        Ci[-1,-1] = np.linalg.norm(anchors[i])**2
        constraints.append(G[i,i] == trace(Ci*X))

    obj = Minimize(trace(G) - 2*sum_entries(mul_elemwise(r, G[:-1,m])))
    prob = Problem(obj, constraints)

    ## Solution
    total = prob.solve()
    #if (print_out):
    #    print('total cost:', total)
    #    print('G:',G.value)
    #    print('X:',X.value)
    return 0
