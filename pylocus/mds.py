#!/usr/bin/env python
# module MDS
import numpy as np
from cvxpy import *

from .basics import eigendecomp


def theta_from_eigendecomp(factor, u):
    theta_hat = np.dot(np.diag(factor[:]), u.T)
    theta_hat = theta_hat[0, :]
    return np.real(theta_hat).reshape((-1,))


def x_from_eigendecomp(factor, u, dim):
    return np.dot(np.diag(factor[:]), u.T)[:dim, :]


def MDS(D, dim, method='simple', theta=False):
    """ recover points from euclidean distance matrix using classic MDS algorithm. 
    """
    N = D.shape[0]
    if method == 'simple':
        d1 = D[0, :]
        G = -0.5 * (D - d1 * np.ones([1, N]).T - (np.ones([N, 1]) * d1).T)
        factor, u = eigendecomp(G, dim)
        if (theta):
            return theta_from_eigendecomp(factor, u)
        else:
            return x_from_eigendecomp(factor, u, dim)
    if method == 'advanced':
        s1T = np.vstack([np.ones([1, N]), np.zeros([N - 1, N])])
        G = -0.5 * np.dot(np.dot((np.identity(N) - s1T.T), D),
                          (np.identity(N) - s1T))
        factor, u = eigendecomp(G, dim)
        if (theta):
            return theta_from_eigendecomp(factor, u)
        else:
            return x_from_eigendecomp(factor, u, dim)
    if method == 'geometric':
        J = np.identity(N) - 1.0 / float(N) * np.ones([N, N])
        G = -0.5 * np.dot(np.dot(J, D), J)
        factor, u = eigendecomp(G, dim)
        if (theta):
            return theta_from_eigendecomp(factor, u)
        else:
            return x_from_eigendecomp(factor, u, dim)
    else:
        print('Unknown method {} in MDS'.format(method))


def superMDS(X0, N, d, **kwargs):
    """ Find the set of points from an edge kernel.
    """
    Om = kwargs.get('Om', None)
    dm = kwargs.get('dm', None)
    if Om is not None and dm is not None:
        KE = kwargs.get('KE', None)
        if KE is not None:
            print('superMDS: KE and Om, dm given. Continuing with Om, dm')
        factor, u = eigendecomp(Om, d)
        uhat = u[:, :d]
        lambdahat = np.diag(factor[:d])
        diag_dm = np.diag(dm)
        Vhat = np.dot(diag_dm, np.dot(uhat, lambdahat))
    elif Om is None or dm is None:
        KE = kwargs.get('KE', None)
        if KE is None:
            raise NameError('Either KE or Om and dm have to be given.')
        factor, u = eigendecomp(KE, d)
        lambda_ = np.diag(factor)
        Vhat = np.dot(u, lambda_)[:, :d]

    C_inv = -np.eye(N)
    C_inv[0, 0] = 1.0
    C_inv[:, 0] = 1.0
    b = np.zeros((C_inv.shape[1], d))
    b[0, :] = X0
    b[1:, :] = Vhat[:N - 1, :]
    Xhat = np.dot(C_inv, b)
    return Xhat, Vhat


def iterativeEMDS(X0, N, d, C, b, max_it=10, print_out=False, **kwargs):
    """ Find the set of points from an edge kernel with geometric constraints, using iterative projection 
    """
    from pylocus.basics import mse, projection
    KE = kwargs.get('KE', None)
    KE_projected = KE.copy()
    d = len(X0)
    for i in range(max_it):
        # projection on constraints
        KE_projected, cost, __ = projection(KE_projected, C, b)
        rank = np.linalg.matrix_rank(KE_projected)

        # rank 2 decomposition
        Xhat_KE, Vhat_KE = superMDS(X0, N, d, KE=KE_projected)
        KE_projected = Vhat_KE.dot(Vhat_KE.T)

        error = mse(C.dot(KE_projected), b)

        if (print_out):
            print('cost={:2.2e},error={:2.2e}, rank={}'.format(
                cost, error, rank))
        if cost < 1e-20 and error < 1e-20 and rank == d:
            if (print_out):
                print('converged after {} iterations'.format(i))
            return Xhat_KE, Vhat_KE
    print('iterativeMDS did not converge!')
    return None, None


def relaxedEMDS(X0, N, d, C, b, KE, print_out=False, lamda=10):
    """ Find the set of points from an edge kernel with geometric constraints, using convex rank relaxation.
    """
    E = C.shape[1]
    X = Semidef(E)

    constraints = [C[i, :] * X == b[i] for i in range(C.shape[0])]

    obj = Minimize(trace(X) + lamda * norm(KE - X))
    prob = Problem(obj, constraints)

    try:
        # CVXOPT is more accurate than SCS, even though slower.
        total_cost = prob.solve(solver='CVXOPT', verbose=print_out)
    except:
        try:
            print('CVXOPT with default cholesky failed. Trying kktsolver...')
            # kktsolver is more robust than default (cholesky), even though slower.
            total_cost = prob.solve(
                solver='CVXOPT', verbose=print_out, kktsolver="robust")
        except:
            try:
                print('CVXOPT with robust kktsovler failed. Trying SCS...')
                # SCS is fast and robust, but inaccurate (last choice).
                total_cost = prob.solve(solver='SCS', verbose=print_out)
            except:
                print('SCS and CVXOPT solver with default and kktsolver failed .')
    if print_out:
        print('status:', prob.status)
    Xhat_KE, Vhat_KE = superMDS(X0, N, d, KE=X.value)
    return Xhat_KE, Vhat_KE


def signedMDS(cdm, W=None):
    """ Find the set of points from a cdm.
    """

    N = cdm.shape[0]

    D_sym = (cdm - cdm.T) / 2

    if W is None:
        x_est = np.mean(D_sym, axis=1)
        return x_est - np.min(x_est)

    W_sub = W[1:, 1:]
    sum_W = np.sum(W[1:, :], axis=1)

    #    A = np.eye(N, N-1, k=-1) - W_sub.astype(np.int) / sum_W[:, None]
    A = np.eye(N - 1) - W_sub.astype(np.int) / 1. / sum_W[:, None]
    d = (np.sum(D_sym[1:, :] * W[1:, :], axis=1) / 1. / sum_W)

    x_est = np.linalg.lstsq(A, d)[0]
    x_est = np.r_[[0], x_est]

    return x_est - np.min(x_est), A, np.linalg.pinv(A)


if __name__ == "__main__":
    print('nothing happens when running this module.')
