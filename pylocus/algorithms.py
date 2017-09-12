#!/usr/bin/env python
# module ALGORITHMS
import numpy as np


def execute_method(method, noisy_edm=None, real_points=None, W=None, **kwargs):
    lamda = 1000
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
            noisy_edm, W, lamda, real_points)
        ### Added to avoid strange "too large to be a matrix" error
        N, d = real_points.shape
        xhat = np.zeros((N, d))
        xhat[:, :] = x_SDRold
    if method == 'ACD':
        X0 = kwargs.get('X0', None)
        xhat, fs, err_edms, err_points = reconstruct_acd(noisy_edm, W=W, X0=X0.copy(),
                                                         real_points=real_points)
    if method == 'dwMDS':
        X0 = kwargs.get('X0', None)
        X_bar = kwargs.get('X_bar', None)
        r = kwargs.get('r', None)
        n = kwargs.get('n', None)
        xhat, costs = reconstruct_dwmds(noisy_edm, X0=X0.copy(), W=W, n=n,
                                        X_bar=X_bar, r=r)
    if method == 'SRLS':
        n = kwargs.get('n', None)
        print('in SRLS:', n)
        xhat = reconstruct_srls(noisy_edm, real_points,
                                indices=range(n), W=W)
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


def reconstruct_emds(edm, Om, real_points, iterative=False, **kwargs):
    """ Reconstruct point set using E(dge)-MDS.
    """
    from .point_set import dm_from_edm
    N = real_points.shape[0]
    d = real_points.shape[1]
    dm = dm_from_edm(edm)
    if not iterative:
        from .mds import superMDS
        Xhat, __ = superMDS(real_points[0, :], N, d, Om=Om, dm=dm)
    else:
        from .mds import iterativeMDS
        C = kwargs.get('C', None)
        b = kwargs.get('b', None)
        if C is None or b is None:
            raise NameError(
                'Need constraints C and b for reconstruct_emds in iterative mode.')
        KE = np.multiply(np.outer(dm, dm), Om)
        Xhat, __ = iterativeMDS(real_points[0, :], N, d, KE=KE, C=C, b=b)
    Y, R, t, c = procrustes(real_points, Xhat, scale=False)
    return Y


def reconstruct_smds(dm, absolute_angles, real_points, W=None):
    """ Reconstruct point set using signed Multidimensional Scaling.
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


def reconstruct_sdp(edm, W, lamda, points, print_out=False):
    """ Reconstruct point set using semi-definite rank relaxation.
    """
    from .edm_completion import semidefinite_relaxation
    edm_complete = semidefinite_relaxation(edm, W, lamda, print_out)
    Xhat = reconstruct_mds(edm_complete, points, method='geometric')
    return Xhat, edm_complete


def reconstruct_srls(edm, real_points, print_out=False, indices=[0], W=None):
    """ Reconstruct point set using S(quared)R(ange)L(east)S(quares) method.
    """
    from .lateration import SRLS, get_lateration_parameters
    Y = real_points.copy()
    for index in indices:
        anchors, w, r2 = get_lateration_parameters(real_points, indices, index,
                                                   edm, W)
        if print_out:
            print('SRLS parameters:',anchors, w, r2)
        srls = SRLS(anchors, w, r2, print_out)
        Y[index, :] = srls
    return Y


def reconstruct_acd(edm, W, X0, real_points, print_out=False, tol=1e-10):
    """ Reconstruct point set using alternating coordinate descent.
    """
    from .point_set import create_from_points, PointSet
    from .distributed_mds import get_step_size, f
    X_k = X0.copy()
    N = X_k.shape[0]
    d = X_k.shape[1]

    # create reference object
    preal = create_from_points(real_points, PointSet)
    # create optimization object
    cd = create_from_points(X_k, PointSet)

    fs = []
    err_edms = []
    err_points = []
    done = False

    coord_n_it = 10

    coordinates_converged = np.ones(N) * coord_n_it
    for sweep_counter in range(100):
        # sweep
        for i in np.where(coordinates_converged > 1)[0]:
            coord_counter = 0
            while coord_counter < coord_n_it:
                coord_counter += 1
                for coord in range(d):
                    delt = get_step_size(i, coord, X_k, edm, W)
                    if len(delt) > 1:
                        ftest = []
                        for de in delt:
                            X_ktest = X_k.copy()
                            X_ktest[i, coord] += de
                            ftest.append(f(X_ktest, edm, W))
                        delt = delt[ftest == min(ftest)]
                    X_k[i, coord] += delt
                    f_this = f(X_k, edm, W)
                    fs.append(f_this)
                    cd.points = X_k
                    cd.init()
                    err_edms.append(np.linalg.norm(cd.edm - edm))
                    err_points.append(np.linalg.norm(X_k - preal.points))
                    if len(fs) > 2:
                        if abs(fs[-1] - fs[-2]) < tol:
                            if (print_out):
                                print(fs[-1])
                                print(fs[-2])
                                print('acd: coordinate converged after {} loops.'.format(
                                    coord_counter))
                            if coord_counter == 1:
                                if coord_counter > coordinates_converged[i]:
                                    print(
                                        'Unexpected behavior: Coordinate converged in more than before')
                            coordinates_converged[i] = coord_counter
                            coord_counter = coord_n_it
                            break
                        else:
                            pass
                            #print('error:',abs(fs[-1] - fs[-2]))
        if (coordinates_converged == 1).all():
            if (print_out):
                print('acd: all coordinates converged after {} sweeps.'.format(
                    sweep_counter))
            return X_k, fs, err_edms, err_points
        else:
            if (print_out):
                print('acd: not yet converged:', coordinates_converged)
    if (print_out):
        print('acd: did not converge after {} sweeps'.format(sweep_counter + 1))
    return X_k, fs, err_edms, err_points


def reconstruct_dwmds(edm, X0, W, r=None, n=None, X_bar=None, max_iter=100, tol=1e-10, print_out=False):
    """ Reconstruct point set using d(istributed)w(eighted) MDS.
    """
    from .basics import get_edm
    from .distributed_mds import get_b, get_Si

    N, d = X0.shape
    if r is None and n is None:
        raise ValueError('either r or n have to be given.')
    elif n is None:
        n = r.shape[0]

    costs = []
    X = X0.copy()

    # don't have to ignore i=j, because W[i,i] is zero.
    a = np.sum(W[:n, :n], axis=1).flatten() + 2 * \
        np.sum(W[:n, n:], axis=1).flatten()
    if r is not None:
        a += r.flatten()
    for k in range(max_iter):
        S = 0
        for i in range(n):
            edm_estimated = get_edm(X)
            bi = get_b(i, edm_estimated, W, edm, n)
            if r is not None and X_bar is not None:
                X[i] = 1 / a[i] * (r[i] * X_bar[i, :] + X.T.dot(bi).flatten())
                Si = get_Si(i, edm_estimated, edm, W, n, r, X_bar[i], X[i])
            else:
                X[i] = 1 / a[i] * X.T.dot(bi).flatten()
                Si = get_Si(i, edm_estimated, edm, W, n)
            S += Si
        costs.append(S)
        if k > 1 and abs(costs[-1] - costs[-2]) < tol:
            if (print_out):
                print('dwMDS: converged after', k)
                print('dwMDS: costs:', costs)
            break
    return X, costs


if __name__ == "__main__":
    print('nothing happens when running this module.')
