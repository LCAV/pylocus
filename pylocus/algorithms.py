#!/usr/bin/env python
# module ALGORITHMS
import numpy as np


def execute_method(method, noisy_edm=None, real_points=None, W=None, **kwargs):
    if method == 'MDS':
        xhat = reconstruct_mds(
            noisy_edm, real_points=real_points, method='geometric')
    if method == 'MDSoptspace':
        xhat = reconstruct_mds(noisy_edm, real_points=real_points,
                               method='geometric', mask=W,
                               completion='optspace', print_out=False)
    if method == 'MDSalternate':
        xhat = reconstruct_mds(noisy_edm, real_points=real_points,
                               method='geometric', mask=W,
                               completion='alternate', print_out=False)
    if method == 'SDR':
        x_SDRold, EDMbest = reconstruct_sdp(
            noisy_edm, W=W, real_points=real_points)
        # Added to avoid strange "too large to be a matrix" error
        N, d = real_points.shape
        xhat = np.zeros((N, d))
        xhat[:, :] = x_SDRold
    if method == 'ACD':
        X0 = kwargs.get('X0', None)
        xhat, costs = reconstruct_acd(noisy_edm, W=W, X0=X0)
    if method == 'dwMDS':
        X0 = kwargs.pop('X0', None)
        xhat, costs = reconstruct_dwmds(noisy_edm, W=W, X0=X0, **kwargs)
    if method == 'SRLS':
        n = kwargs.get('n', None)
        rescale = kwargs.get('rescale', False)
        xhat = reconstruct_srls(noisy_edm, real_points,
                                n=n, W=W, rescale=rescale)
    return xhat


def classical_mds(D):
    from .mds import MDS
    return MDS(D, 1, 'geometric')


def procrustes(anchors, X, scale=True, print_out=False):
    """ Fit X to anchors by applying optimal translation, rotation and reflection.

    Given m >= d anchor nodes (anchors in R^(m x d)), return transformation
    of coordinates X (output of EDM algorithm) optimally matching anchors in least squares sense.

    :param anchors: Matrix of shape m x d, where m is number of anchors, d is dimension of setup.
    :param X: Matrix of shape N x d, where the last m points will be used to find fit with the anchors. 
    :param scale: set to True if the point set should be scaled to match the anchors.
    
    :return: the transformed vector X, the rotation matrix, translation vector, and scaling factor.
    """
    def centralize(X):
        n = X.shape[0]
        ones = np.ones((n, 1))
        return X - np.multiply(1 / n * np.dot(ones.T, X), ones)
    m = anchors.shape[0]
    N, d = X.shape
    assert m >= d, 'Have to give at least d anchor nodes.'
    X_m = X[N - m:, :]
    ones = np.ones((m, 1))

    mux = 1 / m * np.dot(ones.T, X_m)
    muy = 1 / m * np.dot(ones.T, anchors)
    sigmax = 1 / m * np.linalg.norm(X_m - mux)**2
    sigmaxy = 1 / m * np.dot((anchors - muy).T, X_m - mux)
    try:
        U, D, VT = np.linalg.svd(sigmaxy)
    except np.LinAlgError:
        print('strange things are happening...')
        print(sigmaxy)
        print(np.linalg.matrix_rank(sigmaxy))
    #this doesn't work and doesn't seem to be necessary! (why?)
    #  S = np.eye(D.shape[0])
    #  if (np.linalg.det(U)*np.linalg.det(VT.T) < 0):
    #  print('switching')
    #  S[-1,-1] = -1.0
    #  else:
    #  print('not switching')
    #  c = np.trace(np.dot(np.diag(D),S))/sigmax
    #  R = np.dot(U, np.dot(S,VT))
    if (scale):
        c = np.trace(np.diag(D)) / sigmax
    else:
        c = np.trace(np.diag(D)) / sigmax
        if (print_out):
            print('Optimal scale would be: {}. Setting it to 1 now.'.format(c))
        c = 1.0
    R = np.dot(U, VT)
    t = muy.T - c * np.dot(R, mux.T)
    X_transformed = (c * np.dot(R, (X - mux).T) + muy.T).T
    return X_transformed, R, t, c


def reconstruct_emds(edm, Om, real_points, method=None, **kwargs):
    """ Reconstruct point set using E(dge)-MDS.
    """
    from .point_set import dm_from_edm
    N = real_points.shape[0]
    d = real_points.shape[1]
    dm = dm_from_edm(edm)
    if method is None:
        from .mds import superMDS
        Xhat, __ = superMDS(real_points[0, :], N, d, Om=Om, dm=dm)
    else:
        C = kwargs.get('C', None)
        b = kwargs.get('b', None)
        if C is None or b is None:
            raise NameError(
                'Need constraints C and b for reconstruct_emds in iterative mode.')
        KE_noisy = np.multiply(np.outer(dm, dm), Om)
        if method == 'iterative':
            from .mds import iterativeEMDS
            Xhat, __ = iterativeEMDS(
                real_points[0, :], N, d, KE=KE_noisy, C=C, b=b)
        elif method == 'relaxed':
            from .mds import relaxedEMDS
            Xhat, __ = relaxedEMDS(
                real_points[0, :], N, d, KE=KE_noisy, C=C, b=b)
        else:
            raise NameError('Undefined method', method)
    Y, R, t, c = procrustes(real_points, Xhat, scale=False)
    return Y


def reconstruct_cdm(dm, absolute_angles, real_points, W=None):
    """ Reconstruct point set from angle and distance measurements, using coordinate difference matrices.
    """
    from pylocus.point_set import dmi_from_V, sdm_from_dmi, get_V
    from pylocus.mds import signedMDS

    N = real_points.shape[0]

    V = get_V(absolute_angles, dm)

    dmx = dmi_from_V(V, 0)
    dmy = dmi_from_V(V, 1)

    sdmx = sdm_from_dmi(dmx, N)
    sdmy = sdm_from_dmi(dmy, N)

    points_x = signedMDS(sdmx, W)
    points_y = signedMDS(sdmy, W)

    Xhat = np.c_[points_x, points_y]
    Y, R, t, c = procrustes(real_points, Xhat, scale=False)
    return Y


def reconstruct_mds(edm, real_points, completion='optspace', mask=None, method='geometric', print_out=False, n=1):
    """ Reconstruct point set using MDS and matrix completion algorithms.
    """
    from .point_set import dm_from_edm
    from .mds import MDS
    N = real_points.shape[0]
    d = real_points.shape[1]
    if mask is not None:
        edm_missing = np.multiply(edm, mask)
        if completion == 'optspace':
            from .edm_completion import optspace
            edm_complete = optspace(edm_missing, d + 2)
        elif completion == 'alternate':
            from .edm_completion import rank_alternation
            edm_complete, errs = rank_alternation(
                edm_missing, d + 2, print_out=False, edm_true=edm)
        else:
            raise NameError('Unknown completion method {}'.format(completion))
        if (print_out):
            err = np.linalg.norm(edm_complete - edm)**2 / \
                np.linalg.norm(edm)**2
            print('{}: relative error:{}'.format(completion, err))
        edm = edm_complete
    Xhat = MDS(edm, d, method, False).T
    Y, R, t, c = procrustes(real_points[n:], Xhat, True)
    #Y, R, t, c = procrustes(real_points, Xhat, True)
    return Y


def reconstruct_sdp(edm, real_points, W=None, print_out=False, lamda=1000, **kwargs):
    """ Reconstruct point set using semi-definite rank relaxation.
    """
    from .edm_completion import semidefinite_relaxation
    edm_complete = semidefinite_relaxation(
        edm, lamda=lamda, W=W, print_out=print_out, **kwargs)
    Xhat = reconstruct_mds(edm_complete, real_points, method='geometric')
    return Xhat, edm_complete


def reconstruct_srls(edm, real_points, W=None, print_out=False, n=1, rescale=False,
                     z=None):
    """ Reconstruct point set using S(quared)R(ange)L(east)S(quares) method.
    """
    from .lateration import SRLS, get_lateration_parameters
    Y = real_points.copy()
    indices = range(n)
    for index in indices:
        anchors, w, r2 = get_lateration_parameters(real_points, indices, index,
                                                   edm, W)
        if print_out:
            print('SRLS parameters:', anchors, w, r2)
        srls = SRLS(anchors, w, r2, rescale, z, print_out)
        if rescale:
            srls = srls[0]  # second element of output is the scale
        Y[index, :] = srls
    return Y


def reconstruct_acd(edm, X0, W=None, print_out=False, tol=1e-10, sweeps=10):
    """ Reconstruct point set using alternating coordinate descent.

    :param X0: Nxd matrix of starting points.
    """

    def get_unique_delta(delta, i, coord, print_out=False):
        delta_unique = delta.copy()
        ftest = []
        for de in delta_unique:
            X_ktest = X_k.copy()
            X_ktest[i, coord] += de
            ftest.append(f(X_ktest, edm, W))
        # choose delta_unique with lowest cost.
        if print_out:
            print(ftest)
            print(delta_unique)
        delta_unique = delta_unique[ftest == min(ftest)]
        if print_out:
            print(delta_unique)
        if len(delta_unique) > 1:
            # if multiple delta_uniques give the same cost, choose the biggest step.
            delta_unique = max(delta_unique)
        return delta_unique

    def sweep():
        for i in range(N):
            loop_point(i)

    def loop_coordinate(coord, i):
        delta = get_step_size(i, coord, X_k, edm, W)
        if len(delta) > 1:
            delta_unique = get_unique_delta(delta, i, coord)
        elif len(delta) == 1:
            delta_unique = delta[0]
        else:
            print('Warning: did not find delta!', delta)
            delta_unique = 0.0
        try:
            X_k[i, coord] += delta_unique
        except:
            get_unique_delta(delta, i, coord, print_out=True)
            raise

        cost_this = f(X_k, edm, W)
        costs.append(cost_this)

        if delta_unique <= tol:
            coordinates_converged[i, coord] += 1
        else:
            if print_out:
                print('======= coordinate {} of {} not yet converged'.format(coord, i))
            coordinates_converged[i, coord] = 0

    def loop_point(i):
        coord_counter = 0
        while not (coordinates_converged[i] >= 2).all():
            if print_out:
                print('==== point {}'.format(i))
            coord_counter += 1
            for coord in range(d):
                if print_out:
                    print('======= coord {}'.format(coord))
                loop_coordinate(coord, i)
            if coord_counter > coord_n_it:
                break
    from .distributed_mds import get_step_size, f

    N, d = X0.shape
    if W is None:
        W = np.ones((N, N)) - np.eye(N)

    X_k = X0.copy()

    costs = []
    coordinates_converged = np.zeros(X_k.shape)
    coord_n_it = 3
    for sweep_counter in range(sweeps):
        if print_out:
            print('= sweep', sweep_counter)
        sweep()
        if (coordinates_converged >= 2).all():
            if (print_out):
                print('acd: all coordinates converged after {} sweeps.'.format(
                    sweep_counter))
            return X_k, costs
    if (print_out):
        print('acd: did not converge after {} sweeps'.format(sweep_counter + 1))
    return X_k, costs


def reconstruct_dwmds(edm, X0, W=None, n=None, r=None, X_bar=None, print_out=False, tol=1e-10, sweeps=100):
    """ Reconstruct point set using d(istributed)w(eighted) MDS.

    :param X0: Nxd matrix of starting points.
    :param n: Number of points of unknown position. The first n points in X0 and edm are considered unknown. 
    """
    from .basics import get_edm
    from .distributed_mds import get_b, get_Si

    N, d = X0.shape
    if r is None and n is None:
        raise ValueError('either r or n have to be given.')
    elif n is None:
        n = r.shape[0]

    if W is None:
        W = np.ones((N, N)) - np.eye(N)

    X_k = X0.copy()

    costs = []
    # don't have to ignore i=j, because W[i,i] is zero.
    a = np.sum(W[:n, :n], axis=1).flatten() + 2 * \
        np.sum(W[:n, n:], axis=1).flatten()
    if r is not None:
        a += r.flatten()
    for k in range(sweeps):
        S = 0
        for i in range(n):
            edm_estimated = get_edm(X_k)
            bi = get_b(i, edm_estimated, W, edm, n)
            if r is not None and X_bar is not None:
                X_k[i] = 1 / a[i] * (r[i] * X_bar[i, :] +
                                     X_k.T.dot(bi).flatten())
                Si = get_Si(i, edm_estimated, edm, W, n, r, X_bar[i], X_k[i])
            else:
                X_k[i] = 1 / a[i] * X_k.T.dot(bi).flatten()
                Si = get_Si(i, edm_estimated, edm, W, n)
            S += Si
        costs.append(S)
        if k > 1 and abs(costs[-1] - costs[-2]) < tol:
            if (print_out):
                print('dwMDS: converged after', k)
            break
    return X_k, costs


if __name__ == "__main__":
    print('nothing happens when running this module.')
