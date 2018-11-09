#!/usr/bin/env python
# module SIMULATION
import numpy as np


def weights_one(N):
    weights = np.ones((N, N))
    weights[range(N), range(N)] = 0.0
    return weights


def weights_zero(N):
    weights = weights_one(N)
    weights[0, 1] = 0.0
    weights[1, 0] = 0.0
    return weights


def weights_linear(N, noise, noise_edm):
    weights = weights_one(N)
    weights[0, 1] = noise_edm / noise
    weights[1, 0] = weights[0, 1]
    return weights


def weights_quadratic(N, noise, noise_edm):
    weights = weights_one(N)
    weights[0, 1] = (noise_edm / noise)**2
    weights[1, 0] = weights[0, 1]
    return weights


def create_noisy_edm(edm, noise, n=None):
    """Create noisy version of edm
    
    Adds symmetric Gaussian noise to non-diagonal elements of EDM (to distances!). 
    The output EDM is ensured to have only positive entries.
    
    :param edm: Original, noiseless EDM.
    :param noise: Standard deviation of Gaussian noise to be added to distances.
    :param n: How many rows/columns to consider. Set to size of edm by default.
    
    :return: Noisy version of input EDM.
    """
    N = edm.shape[0]
    if n is None:
        n = N
    found = False
    max_it = 100
    i = 0
    while not found:
        i += 1
        dm = np.sqrt(edm) + np.random.normal(scale=noise, size=edm.shape)
        dm = np.triu(dm)
        edm_noisy = np.power(dm + dm.T, 2)
        edm_noisy[range(N), range(N)] = 0.0
        edm_noisy[n:, n:] = edm[n:, n:]
        if (edm_noisy >= 0).all():
            found = True
        if i > max_it:
            print('create_noisy_edm: last EDM', edm_noisy)
            raise RuntimeError(
                'Could not generate all positive edm in {} iterations.'.format(max_it))
    return edm_noisy


def create_mask(N, method='all', nmissing=0):
    """ Create weight mask according to method.

    :param N: Dimension of square weight matrix.
    :param method: Method to use (default: 'all').
    - none: no missing entries (only diagonal is set to 0 for dwMDS)
    - first: only randomly delete measurements to first point (zeros in first row/column of matrix)
    - all: randomly delete measurements in whole matrix
    :param nmissing: Number of deleted measurements, used by methods 'first' and 'all'

    :return: Binary weight mask.
    :rtype: numpy.ndarray
    """

    weights = np.ones((N, N))
    weights[range(N), range(N)] = 0

    if method == 'none':
        return weights

    # create indices object to choose from
    elif method == 'all':
        all_indices = np.triu_indices(N, 1)
    elif method == 'first':
        all_indices = [np.zeros(N - 1).astype(np.int),
                       np.arange(1, N).astype(np.int)]
    ntotal = len(all_indices[0])
    # randomly choose from indices and set to 0
    choice = np.random.choice(ntotal, nmissing, replace=False)
    chosen = [all_indices[0][choice], all_indices[1][choice]]
    weights[chosen] = 0
    weights[chosen[1], chosen[0]] = 0
    return weights


def create_weights(N, method='one', noise=0.0, noise_edm=None):
    if method == 'one':
        return weights_one(N)
    elif method == 'zero':
        return weights_zero(N)
    elif method == 'linear':
        return weights_linear(N, noise, noise_edm)
    elif method == 'quadratic':
        return weights_quadratic(N, noise, noise_edm)
    else:
        raise NameError("Unknown method", method)



if __name__ == "__main__":
    print('nothing happens when running this module.')
