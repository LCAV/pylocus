#!/usr/bin/env python
# module SEMIDEFINITE

## SDP Problem
from cvxpy import *
import numpy as np


def reconstruct_sdp(EDM_noisy, W, lamda, points, print_out=False):
    from algorithms import reconstruct_mds
    def kappa(gram):
        n = len(gram)
        e = np.ones(n)
        return np.outer(np.diag(gram), e) + np.outer(e, np.diag(gram).T) - 2 * gram

    def kappa_cvx(gram, n):
        e = np.ones((n, 1))
        return diag(gram) * e.T + e * diag(gram).T - 2 * gram
        from algorithms import reconstruct_mds

    n = EDM_noisy.shape[0]
    V = np.c_[-np.ones(n - 1) / np.sqrt(n), np.eye(n - 1) -
              np.ones([n - 1, n - 1]) / (n + np.sqrt(n))].T

    # Creates a n-1 by n-1 positive semidefinite variable.
    H = Semidef(n - 1)
    G = V * H * V.T  # * is overloaded
    EDM = kappa_cvx(G, n)

    obj = Maximize(trace(H) - lamda * norm(mul_elemwise(W, (EDM - EDM_noisy))))
    prob = Problem(obj)

    ## Solution
    total = prob.solve()
    if (print_out):
        print('total cost:', total)
    

    Gbest = V * H.value * V.T
    EDMbest = kappa(Gbest)

    # TODO why do these two not sum up to the objective?
    if (print_out):
        print('trace of H:', np.trace(H.value))
        print('other cost:', lamda * norm(mul_elemwise(W, (EDMbest - EDM_noisy))).value)

    Ubest, Sbest, Vbest = np.linalg.svd(Gbest)
    Xhat = reconstruct_mds(EDMbest, points, method='geometric')
    return Xhat, EDMbest
    # TODO: why does this not work?
    #from basics import eigendecomp
    #factor, u = eigendecomp(Gbest, d)
    #Xhat = np.diag(factor).dot(u.T)[:d]

if __name__ == "__main__":
    print('This module contains functions for semidefinite programming approach of edm-based algorithms.')
