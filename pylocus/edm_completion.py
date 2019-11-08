#!/usr/bin/env python
# module EDM_COMPLETION
import numpy as np

import cvxpy as cp

from pylocus.basics import get_edm


def optspace(edm_missing, rank, niter=500, tol=1e-6, print_out=False):
    """Complete and denoise EDM using OptSpace algorithm.

    Uses OptSpace algorithm to complete and denoise EDM. The problem being solved is
    X,S,Y = argmin_(X,S,Y) || W ° (D - XSY') ||_F^2

    :param edm_missing: EDM with 0 where no measurement was taken.
    :param rank: expected rank of complete EDM.
    :param niter, tol: see opt_space module for description.

    :return: Completed matrix.
    """
    from .opt_space import opt_space
    N = edm_missing.shape[0]
    X, S, Y, __ = opt_space(edm_missing, r=rank, niter=niter,
                            tol=tol, print_out=print_out)
    edm_complete = X.dot(S.dot(Y.T))
    edm_complete[range(N), range(N)] = 0.0
    return edm_complete


def rank_alternation(edm_missing, rank, niter=50, print_out=False, edm_true=None):
    """Complete and denoise EDM using rank alternation.

    Iteratively impose rank and strucutre to complete marix entries

    :param edm_missing: EDM with 0 where no measurement was taken.
    :param rank: expected rank of complete EDM.
    :param niter: maximum number of iterations.
    :param edm: if given, the relative EDM error is tracked. 

    :return: Completed matrix and array of errors (empty if no true edm is given).
    The matrix is of the correct structure, but might not have the right measured entries.

    """
    from pylocus.basics import low_rank_approximation
    errs = []
    N = edm_missing.shape[0]
    edm_complete = edm_missing.copy()
    edm_complete[edm_complete == 0] = np.mean(edm_complete[edm_complete > 0])
    for i in range(niter):
        # impose matrix rank
        edm_complete = low_rank_approximation(edm_complete, rank)

        # impose known entries
        edm_complete[edm_missing > 0] = edm_missing[edm_missing > 0]

        # impose matrix structure
        edm_complete[range(N), range(N)] = 0.0
        edm_complete[edm_complete < 0] = 0.0
        edm_complete = 0.5 * (edm_complete + edm_complete.T)

        if edm_true is not None:
            err = np.linalg.norm(edm_complete - edm_true)
            errs.append(err)
    return edm_complete, errs


def semidefinite_relaxation(edm_missing, lamda, W=None, print_out=False, **kwargs):
    """Complete and denoise EDM using semidefinite relaxation.

    Returns solution to the relaxation of the following problem: 
        D = argmin || W * (D - edm_missing) || 
            s.t. D is EDM
    where edm_missing is measured matrix, W is a weight matrix, and * is pointwise multiplication. 

    Refer to paper "Euclidean Distance Matrices - Essential Theory, Algorithms and Applications", 
    Algorithm 5, for details. (https://www.doi.org/%2010.1109/MSP.2015.2398954)

    :param edm_missing: EDM with 0 where no measurement was taken. 
    :param lamda: Regularization parameter.  
    :param W: Optional mask. If no mask is given, a binary mask is created based on missing elements of edm_missing. 
              If mask is given 
    :param kwargs: more options passed to the solver. See cvxpy documentation for all options. 
    """ 
    from .algorithms import reconstruct_mds

    def kappa(gram):
        n = len(gram)
        e = np.ones(n)
        return np.outer(np.diag(gram), e) + np.outer(e, np.diag(gram).T) - 2 * gram

    def kappa_cvx(gram, n):
        e = np.ones((n, 1))
        return cp.reshape(cp.diag(gram), (n, 1)) * e.T + e * cp.reshape(cp.diag(gram), (1, n)) - 2 * gram

    method = kwargs.pop('method', 'maximize')
    options = {'solver': 'CVXOPT'}
    options.update(kwargs)

    if W is None:
        W = (edm_missing > 0)
    else:
        W[edm_missing == 0] = 0.0

    n = edm_missing.shape[0]
    V = np.c_[-np.ones((n - 1, 1)) / np.sqrt(n), np.eye(n - 1) -
              np.ones((n - 1, n - 1)) / (n + np.sqrt(n))].T

    H = cp.Variable((n - 1, n - 1), PSD=True)
    G = V * H * V.T  # * is overloaded
    edm_optimize = kappa_cvx(G, n)

    if method == 'maximize':
        obj = cp.Maximize(cp.trace(H) - lamda *
                       cp.norm(cp.multiply(W, (edm_optimize - edm_missing)), p=1))
    # TODO: add a reference to paper where "minimize" is used instead of maximize. 
    elif method == 'minimize':
        obj = cp.Minimize(cp.trace(H) + lamda *
                       norm(cp.multiply(W, (edm_optimize - edm_missing)), p=1))

    prob = cp.Problem(obj)

    total = prob.solve(**options)
    if print_out:
        print('total cost:', total)
        print('SDP status:', prob.status)

    if H.value is not None:
        Gbest = V.dot(H.value).dot(V.T)
        if print_out:
            print('eigenvalues:', np.sum(np.linalg.eigvals(Gbest)[2:]))
        edm_complete = kappa(Gbest)
    else:
        edm_complete = edm_missing

    if (print_out):
        if H.value is not None:
            print('cp.trace of H:', cp.trace(H.value))
        print('other cost:', lamda *
              norm(cp.multiply(W, (edm_complete - edm_missing)), p=1).value)

    return np.array(edm_complete)


def completion_acd(edm, X0, W=None, tol=1e-6, sweeps=3):
    """ Complete an denoise EDM using alternating decent. 

    The idea here is to simply run reconstruct_acd for a few iterations, 
    yieding a position estimate, which can in turn be used 
    to get a completed and denoised edm. 

    :param edm: noisy matrix (NxN) 
    :param X0: starting points (Nxd) 
    :param W: optional weight matrix. 
    :param tol: Stopping criterion of iterative algorithm.
    :param sweeps: Maximum number of sweeps. 
    """ 
    from .algorithms import reconstruct_acd
    Xhat, costs = reconstruct_acd(edm, X0, W, tol=tol, sweeps=sweeps)
    return get_edm(Xhat)


def completion_dwmds(edm, X0, W=None, tol=1e-10, sweeps=100):
    """ Complete an denoise EDM using dwMDS. 

    The idea here is to simply run reconstruct_dwmds for a few iterations, 
    yieding a position estimate, which can in turn be used 
    to get a completed and denoised edm. 

    :param edm: noisy matrix (NxN) 
    :param X0: starting points (Nxd) 
    :param W: optional weight matrix. 
    :param tol: Stopping criterion of iterative algorithm.
    :param sweeps: Maximum number of sweeps. 
    """ 
    from .algorithms import reconstruct_dwmds
    Xhat, costs = reconstruct_dwmds(edm, X0, W, n=1, tol=tol, sweeps=sweeps)
    return get_edm(Xhat)


if __name__ == "__main__":
    print('nothing happens when running this module.')
