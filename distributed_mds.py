#!/usr/bin/env python
# module DISTRIBUTED_MDS
import numpy as np

def reconstruct_dwmds(edm_measured, X, W, r=None, n=None, X_bar=None, max_iter=100, tol=1e-10):
    from basics import get_edm

    def get_b(i, edm_est, W, edm_measured, n):
        b = np.zeros((edm_est.shape[0], 1))
        sum_bi = 0
        for j in range(edm_est.shape[0]):
            wij = W[i, j]
            dij = edm_est[i, j]**0.5
            delta_ij = edm_measured[i, j]**0.5
            if j != i:
                rij = delta_ij / dij
            if j < n and j != i:
                b[j] = wij * (1 - rij)
                sum_bi += wij * rij
            elif j >= n:
                b[j] = 2 * wij * (1 - rij)
                sum_bi += 2 * wij * rij
        b[i] = sum_bi
        return b

    def get_Si(i, edm_est, edm_measured, W, r=None, Xbari=None, Xi=None):
        if not all(v is None for v in (r, Xbari, Xi)) and not all(v is not None for v in (r, Xbari, Xi)):
                raise ValueError(
                    'All or none of r, Xbari, Xi have to be given.')
        matrix = np.multiply(W, (edm_est**0.5 - edm_measured**0.5)**2.0)
        # don't have to ignore i=j, because W[i,i] is zero.
        first = np.sum(matrix[i, :n])
        second = 2 * np.sum(matrix[i, n:])
        Si = first + second
        if r is not None:
            Si += r[i] * np.linalg.norm(Xbari - Xi)**2
        return Si

    N, d = X.shape
    if r is None and n is None:
        raise ValueError('either r or n have to be given.')
    elif n is None:
        n = r.shape[0]

    costs = []
    # don't have to ignore i=j, because W[i,i] is zero.
    a = np.sum(W[:n, :n], axis=1).flatten() + 2 * \
        np.sum(W[:n, n:], axis=1).flatten()
    if r is not None:
        a += r.flatten()
    for k in range(max_iter):
        S = 0
        for i in range(n):
            edm_est = get_edm(X)
            bi = get_b(i, edm_est, W, edm_measured, n)
            if r is not None and X_bar is not None:
                X[i] = 1 / a[i] * (r[i] * X_bar[i, :] + X.T.dot(bi).flatten())
                Si = get_Si(i, edm_est, edm_measured, W, r, X_bar[i], X[i])
            else:
                X[i] = 1 / a[i] * X.T.dot(bi).flatten()
                Xi = get_Si(i, edm_est, edm_measured, W)
                Si = get_Si(i, edm_est, edm_measured, W)
            S += Si
        costs.append(S)
        if k > 1 and abs(costs[-1] - costs[-2]) < tol:
            print('converged after', k)
            break
    return X, costs

if __name__ == "__main__":
    print('This module contains functions for distributed MDS')
