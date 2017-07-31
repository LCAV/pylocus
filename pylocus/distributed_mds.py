#!/usr/bin/env python
# module DISTRIBUTED_MDS
import numpy as np

# ACD


def get_step_size(i, coord, X_k, D, W, print_out=False):
    def grad_f_i_x(i, coord, X_k, D, W, print_out):
        '''
        Returns gradient of f_i with respect to delta X_k at coord.

        '''
        if (print_out):
            pass
        N = D.shape[0]
        other = np.delete(np.arange(X_k.shape[1]), coord)
        a0 = a1 = a2 = a3 = 0
        for j in np.where(W[i, :] != 0.0)[0]:
            beta = X_k[i, coord] - X_k[j, coord]
            alpha = np.linalg.norm(X_k[i, :] - X_k[j, :])**2 - D[i, j]
            a0 += 4 * W[i, j] * alpha * beta
            a1 += 4 * W[i, j] * (2 * (beta**2) + alpha**2)
            a2 += 4 * W[i, j] * 3 * beta
            a3 += 4 * W[i, j]  # multiplies delta^3
        poly = np.polynomial.Polynomial((a0, a1, a2, a3))
        return poly
    # Find roots of grad_f_x, corresponding to zero of gradient.
    poly = grad_f_i_x(i, coord, X_k, D, W, print_out)
    roots = poly.roots()
    delta = np.real(roots[np.isreal(roots)])
    if (print_out):
        from .plots_cti import plot_cost_function
        deltas = np.linspace(delta - 1.0, delta + 1.0, 100)
        fs = []
        for delta_x in deltas:
            X_delta = X_k.copy()
            X_delta[i, coord] += delta_x
            fs.append(f(X_delta, D, W))
        X_0 = X_k[i, coord]
        x_delta = X_k[i, coord] + delta[0]
        names = ['x', 'y', 'z']
        plot_cost_function(deltas, X_0, x_delta, fs,  names[coord])
    return delta


def f(X_k, D, W):
    def f_i(i, X_k, D, W):
        N = D.shape[0]
        sum_ = 0
        for j in range(N):
            sum_ += W[i, j] * \
                (np.linalg.norm(X_k[i, :] - X_k[j, :])**2 - D[i, j])**2
        return sum_
    N = D.shape[0]
    sum_ = 0
    for i in range(N):
        sum_ += f_i(i, X_k, D, W)
    return sum_

# dwMDS


def get_b(i, edm_est, W, edm_measured, n):
    b = np.zeros((edm_est.shape[0], 1))
    sum_bi = 0
    for j in range(edm_est.shape[0]):
        wij = W[i, j]
        dij = edm_est[i, j]**0.5 if edm_est[i, j] > 0 else 0.
        delta_ij = edm_measured[i, j]**0.5 if edm_measured[i, j] > 0 else 0.
        if j != i:
            rij = delta_ij / dij if dij else 0.
        if j < n and j != i:
            b[j] = wij * (1 - rij)
            sum_bi += wij * rij
        elif j >= n:
            b[j] = 2 * wij * (1 - rij)
            sum_bi += 2 * wij * rij
    b[i] = sum_bi
    return b


def get_Si(i, edm_est, edm_measured, W, n, r=None, Xbari=None, Xi=None):
    if not all(v is None for v in (r, Xbari, Xi)) and not all(v is not None for v in (r, Xbari, Xi)):
            raise ValueError(
                'All or none of r, Xbari, Xi have to be given.')
    d_est = np.zeros_like(edm_est)
    d_measured = np.zeros_like(edm_measured)
    d_est[edm_est > 0] = edm_est[edm_est > 0]**0.5
    d_measured[edm_measured > 0] = edm_measured[edm_measured > 0]**0.5
    matrix = np.multiply(W, (d_est - d_measured)**2.0)
    # don't have to ignore i=j, because W[i,i] is zero.
    first = np.sum(matrix[i, :n])
    second = 2 * np.sum(matrix[i, n:])
    Si = first + second
    if r is not None:
        Si += r[i] * np.linalg.norm(Xbari - Xi)**2
    return Si


if __name__ == "__main__":
    print('This module contains functions for distributed MDS')
