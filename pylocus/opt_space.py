#!/usr/bin/env python
# module OPT_SPACE
"""
Python implementation of OptSpace algorithm for matrix completion.

Author: golnoosh
Extended by: frederike
"""

import numpy as np
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.linalg import svds
from numpy.matlib import repmat
import copy


def opt_space(M_E, r=None, niter=50, tol=1e-6, print_out=False):
    '''
    Implementation of the OptSpace matrix completion algorithm.
    An algorithm for Matrix Reconstruction from a partially revealed set.
    Sparse treatment of matrices are removed because of indexing problems in Python.
    Args:
        M_E:    2D numpy array; The partially revealed matrix.
                Matrix with zeroes at the unrevealed indices.
        r:      The rank to be used for reconstruction. If left empty, the rank is guessed in the program.
        niter:  Maximum number of iterations.
        tol:    Stop iterations if norm((XSY' - M_E) * E, 'fro') / sqrt(|E|) < tol, where
                E_{ij} = 1 if M_{ij} is revealed and zero otherwise,
                |E| is the size of the revealed set.
    Returns: The following
        X:      A M_E.shape[0]xr numpy array
        S:      An rxr numpy array
        Y:      A M_E.shape[1]xr numpy matrix such that M_hat = X*S*Y'
        errs:   A vector containing norm((XSY' - M_E) * E, 'fro') / sqrt(|E|) at each iteration.
    '''
    n, m = M_E.shape
    # construct the revealed set
    E = np.zeros(M_E.shape)
    E[np.nonzero(M_E)] = 1
    eps = np.count_nonzero(E) / np.sqrt(m * n)

    if r is None:
        print('Rank not specified. Trying to guess ...')
        r = guess_rank(M_E)
        print('Using Rank : %d' % r)

    m0 = 10000
    rho = 0

    rescal_param = np.sqrt(np.count_nonzero(
        E) * r / np.linalg.norm(M_E, ord='fro')**2)
    M_E = M_E * rescal_param

    if print_out:
        print('Trimming ...')

    M_Et = copy.deepcopy(M_E)
    d = E.sum(0)
    d_ = np.mean(d)

    for col in range(m):
        if E[:, col].sum() > 2 * d_:
            nonzero_ind_list = np.nonzero(E[:, col])
            p = np.random.permutation(len(nonzero_ind_list))
            M_Et[nonzero_ind_list[p[np.ceil(2 * d_):]], col] = 0

    d = E.sum(1)
    d_ = np.mean(d)
    for row in range(n):
        if E[row, :].sum() > 2 * d_:
            nonzero_ind_list = np.nonzero(E[row, :])
            p = np.random.permutation(len(nonzero_ind_list))
            M_Et[nonzero_ind_list[row, p[np.ceil(2 * d_):]]] = 0

    if print_out:
        print('Sparse SVD ...')

    X0, S0, Y0 = svds_descending(M_Et, r)
    del M_Et

    # Initial Guess
    X0 = X0 * np.sqrt(n)
    Y0 = Y0 * np.sqrt(m)
    S0 = S0 / eps

    if print_out:
        print('Iteration\tFit Error')

    # Gradient Descent
    X = copy.deepcopy(X0)
    Y = copy.deepcopy(Y0)
    S = getoptS(X, Y, M_E, E)

    errs = [None] * (niter + 1)
    errs[0] = np.linalg.norm(
        (M_E - np.dot(np.dot(X, S), Y.T)) * E, ord='fro') / np.sqrt(np.count_nonzero(E))

    if print_out:
        print('0\t\t\t%e' % errs[0])

    for i in range(niter):
        # Compute the Gradient
        W, Z = gradF_t(X, Y, S, M_E, E, m0, rho)

        # Line search for the optimum jump length
        t = getoptT(X, W, Y, Z, S, M_E, E, m0, rho)
        X = X + t * W
        Y = Y + t * Z
        S = getoptS(X, Y, M_E, E)

        # Compute the distortion
        errs[i + 1] = np.linalg.norm((M_E - np.dot(np.dot(X, S), Y.T))
                                     * E, ord='fro') / np.sqrt(np.count_nonzero(E))
        if print_out:
            print('%d\t\t\t%e' % (i + 1, errs[i + 1]))

        if abs(errs[i + 1] - errs[i]) < tol:
            break
    S = S / rescal_param

    return X, S, Y, errs


def svds_descending(M, k):
    '''
    In contrast to MATLAB, numpy's svds() arranges the singular
    values in ascending order. In order to have matching codes,
    we wrap it around by a function which re-sorts the singular
    values and singular vectors.
    Args:
        M: 2D numpy array; the matrix whose SVD is to be computed.
        k: Number of singular values to be computed.

    Returns:
        u, s, vt = svds(M, k=k)
    '''
    u, s, vt = svds(M, k=k)
    # reverse columns of u
    u = u[:, ::-1]
    # reverse s
    s = s[::-1]
    # reverse rows of vt
    vt = vt[::-1, :]
    return u, np.diag(s), vt.T


def guess_rank(M_E):
    '''Guess the rank of the incomplete matrix '''
    n, m = M_E.shape
    epsilon = np.count_nonzero(M_E) / np.sqrt(m * n)
    _, S0, _ = svds_descending(M_E, min(100, max(M_E.shape) - 1))
    S0 = np.diag(S0)

    S1 = S0[:-1] - S0[1:]
    S1_ = S1 / np.mean(S1[-10:])
    r1 = 0
    lam = 0.05
    cost = [None] * len(S1_)
    while r1 <= 0:
        for idx in range(len(S1_)):
            cost[idx] = lam * max(S1_[idx:]) + idx
        i2 = np.argmin(cost)
        r1 = np.max(i2)
        lam += 0.05

    cost = [None] * (len(S0) - 1)
    for idx in range(len(S0) - 1):
        cost[idx] = (S0[idx + 1] + np.sqrt(idx * epsilon)
                     * S0[0] / epsilon) / S0[idx]
    i2 = np.argmin(cost)
    r2 = np.max(i2 + 1)
    r = max([r1, r2])

    return r


def F_t(X, Y, S, M_E, E, m0, rho):
    ''' Compute the distortion '''
    r = X.shape[1]
    out1 = (((np.dot(np.dot(X, S), Y.T) - M_E) * E)**2).sum() / 2
    out2 = rho * G(Y, m0, r)
    out3 = rho * G(X, m0, r)

    return out1 + out2 + out3


def G(X, m0, r):
    z = (X**2).sum(1) / (2 * m0 * r)
    y = np.exp((z - 1)**2) - 1
    y[z < 1] = 0

    return sum(y)


def gradF_t(X, Y, S, M_E, E, m0, rho):
    ''' Compute the gradient.
    '''
    n, r = X.shape
    m, r = Y.shape

    XS = np.dot(X, S)
    YS = np.dot(Y, S.T)
    XSY = np.dot(XS, Y.T)

    Qx = np.dot(np.dot(X.T, ((M_E - XSY) * E)), YS) / n
    Qy = np.dot(np.dot(Y.T, ((M_E - XSY) * E).T), XS) / m

    W = np.dot((XSY - M_E) * E, YS) + np.dot(X, Qx) + rho * Gp(X, m0, r)
    Z = np.dot(((XSY - M_E) * E).T, XS) + np.dot(Y, Qy) + rho * Gp(Y, m0, r)

    return W, Z


def Gp(X, m0, r):
    z = (X**2).sum(1) / (2 * m0 * r)
    z = 2 * np.exp((z - 1)**2) * (z - 1)
    z[z < 0] = 0

    return X * repmat(z, r, 1).T / (m0 * r)


def getoptS(X, Y, M_E, E):
    ''' Find Sopt given X, Y
    '''
    n, r = X.shape
    C = np.dot(np.dot(X.T, M_E), Y)
    C = C.flatten()

    A = np.zeros((r * r, r * r))
    for i in range(r):
        for j in range(r):
            ind = j * r + i
            temp = np.dot(
                np.dot(X.T, np.dot(X[:, i, None], Y[:, j, None].T) * E), Y)
            A[:, ind] = temp.flatten()

    S = np.linalg.solve(A, C)

    return np.reshape(S, (r, r)).T


def getoptT(X, W, Y, Z, S, M_E, E, m0, rho):
    ''' Perform line search
    '''
    iter_max = 20
    norm2WZ = np.linalg.norm(W, ord='fro')**2 + np.linalg.norm(Z, ord='fro')**2
    f = np.zeros(iter_max + 1)
    f[0] = F_t(X, Y, S, M_E, E, m0, rho)

    t = -1e-1
    for i in range(iter_max):
        f[i + 1] = F_t(X + t * W, Y + t * Z, S, M_E, E, m0, rho)

        if f[i + 1] - f[0] <= 0.5 * t * norm2WZ:
            return t
        t /= 2
    return t
