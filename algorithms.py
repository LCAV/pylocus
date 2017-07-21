#!/usr/bin/env python
# module ALGORITHMS
import numpy as np


def classical_mds(D):
    from mds import MDS
    return MDS(D, 1, 'geometric')


def procrustes(anchors, X, scale=True):
    '''
    Given m > d anchor nodes (anchors in R^(m x d)), return transformation
    of coordinates X (output of EDM algorithm) 
    optimally matching anchors in least squares sense.
    '''
    def centralize(X):
        n = X.shape[0]
        ones = np.ones((n, 1))
        return X - np.multiply(1 / n * np.dot(ones.T, X), ones)
    m = anchors.shape[0]
    N = X.shape[0]
    X_m = X[:m, :]
    ones = np.ones((m, 1))

    mux = 1 / m * np.dot(ones.T, X_m)
    muy = 1 / m * np.dot(ones.T, anchors)
    sigmax = 1 / m * np.linalg.norm(X_m - mux)**2
    sigmaxy = 1 / m * np.dot((anchors - muy).T, X_m - mux)
    U, D, VT = np.linalg.svd(sigmaxy)
    #S = np.eye(D.shape[0])
    #this doesn't work and doesn't seem to be necessary! (why?)
    #if (np.linalg.det(U)*np.linalg.det(VT.T) < 0):
    #print('switching')
    #S[-1,-1] = -1.0
    #c = np.trace(np.dot(np.diag(D),S))/sigmax
    #R = np.dot(U, np.dot(S,VT))
    if (scale):
        c = np.trace(np.diag(D)) / sigmax
    else:
        c = np.trace(np.diag(D)) / sigmax
        if abs(c - 1) > 1e-10:
            print('scale not equal to 1: {}. Setting it to 1 now.'.format(c))
        c = 1.0
    R = np.dot(U, VT)
    #t = np.dot(muy - c*np.dot(R, mux))
    t = muy.T - c * np.dot(R, mux.T)
    X_transformed2 = c * np.dot(R, X.T) + t
    X_transformed = (c * np.dot(R, (X - mux).T) + muy.T).T
    return X_transformed, R, t, c


def reconstruct_emds(edm, Om, real_points):
    """
    Edge-MDS using distances and angles.
    """
    from mds import superMDS
    N = real_points.shape[0]
    d = real_points.shape[1]
    dm = dm_from_edm(edm)
    Xhat, __ = superMDS(Om, dm, real_points[0, :], N, d)
    Y, R, t, c = procrustes(real_points, Xhat, True)
    return Y


def reconstruct_mds(edm, real_points, completion='optspace', mask=None, method='geometric', print_out=False):
    from point_configuration import dm_from_edm
    from mds import MDS
    N = real_points.shape[0]
    d = real_points.shape[1]
    if mask is not None:
        edm_missing = np.multiply(edm, mask)
        if completion == 'optspace':
            from edm_completion import optspace
            edm_complete = optspace(edm_missing, d + 2)
        elif completion == 'alternate':
            from edm_completion import rank_alternation
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
    Y, R, t, c = procrustes(real_points[:-1], Xhat, True)
    #Y, R, t, c = procrustes(real_points, Xhat, True)
    return Y


def reconstruct_sdp(edm, W, lamda, points, print_out=False):
    from edm_completion import semidefinite_relaxation
    edm_complete = semidefinite_relaxation(edm, W, lamda, print_out)
    Xhat = reconstruct_mds(edm_complete, points, method='geometric')
    return Xhat, edm_complete


def reconstruct_srls(edm, real_points, print_out=False, indices=[-1], W=None):
    from lateration import SRLS
    Y = real_points.copy()
    for index in indices:
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
        srls = SRLS(anchors, w, r2, print_out)
        Y[index, :] = srls
    return Y


def reconstruct_acd(edm, W, X_0, real_points, print_out=False,):
    from point_configuration import create_from_points, PointConfiguration
    from distributed_mds import get_step_size, f
    X_k = X_0.copy()
    N = X_k.shape[0]
    d = X_k.shape[1]

    # create reference object
    preal = create_from_points(real_points, PointConfiguration)
    # create optimization object
    cd = create_from_points(X_k, PointConfiguration)

    if print_out:
        print('=======initialization=======')
        print('---mds---- edm    ', np.linalg.norm(cd.edm - edm))
        print('---real--- edm    ', np.linalg.norm(cd.edm - preal.edm))
        print('---real--- points ', np.linalg.norm(X_k - preal.points))
        print('cost function:', f(X_k, edm, W))
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
                        if abs(fs[-1] - fs[-2]) < 1e-3:
                            if (print_out):
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
            print('======= step {} ======='.format(sweep_counter))
            print('---mds---- edm    ', np.linalg.norm(cd.edm - edm))
            print('---real--- edm    ', np.linalg.norm(cd.edm - preal.edm))
            print('---real--- points ', np.linalg.norm(X_k - preal.points))
            print('cost function:', f(X_k, edm, W))
    if (print_out):
        print('acd: did not converge after {} sweeps'.format(sweep_counter + 1))
    return X_k, fs, err_edms, err_points


def reconstruct_dwmds(edm, X0, W, r=None, n=None, X_bar=None, max_iter=100, tol=1e-10):
    from basics import get_edm
    from distributed_mds import get_b, get_Si

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
            #print('converged after', k)
            break
    return X, costs


if __name__ == "__main__":
    print('nothing happens when running this module.')
