#!/usr/bin/env python
# module ALGORITHMS

import numpy as np
import matplotlib.pyplot as plt
from math import pi, floor, cos, sin
from basics import rmse, eigendecomp, assert_print, assert_all_print


def MDS(D, dim, method='simple', theta=True):
    N = D.shape[0]
    def theta_from_eigendecomp(factor, u):
        theta_hat = np.dot(np.diag(factor[:]), u.T)
        theta_hat = theta_hat[0, :]
        return np.real(theta_hat).reshape((-1,))
    if method == 'simple':
        d1 = D[0, :]
        G = -0.5 * (D - d1 * np.ones([1, N]).T - (np.ones([N, 1]) * d1).T)
        factor, u = eigendecomp(G, dim)
        if (theta):
            return theta_from_eigendecomp(factor, u)
        else:
            return np.dot(np.diag(factor[:]), u.T)[:dim, :]
    if method == 'advanced':
        s1T = np.vstack([np.ones([1, N]), np.zeros([N - 1, N])])
        G = -0.5 * np.dot(np.dot((np.identity(N) - s1T.T), D),
                          (np.identity(N) - s1T))
        factor, u = eigendecomp(G, dim)
        if (theta):
            return theta_from_eigendecomp(factor, u)
        else:
            return np.dot(np.diag(factor[:]), u.T)[:dim, :]
    if method == 'geometric':
        J = np.identity(N) - 1.0 / float(N) * np.ones([N, N])
        G = -0.5 * np.dot(np.dot(J, D), J)
        factor, u = eigendecomp(G, dim)
        if (theta):
            return theta_from_eigendecomp(factor, u)
        else:
            return np.dot(np.diag(factor[:]), u.T)[:dim, :]
    else:
        print('Unknown method {} in MDS'.format(method))


def classical_mds(D):
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


def super_mds(Om, dm, X0, N, d):
    from basics import eigendecomp
    factor, u = eigendecomp(Om, d)
    uhat = u[:, :d]
    lambdahat = np.diag(factor[:d])
    diag_dm = np.diag(dm)
    Vhat = np.dot(diag_dm, np.dot(uhat, lambdahat))
    C_inv = -np.eye(N)
    C_inv[0, 0] = 1.0
    C_inv[:, 0] = 1.0
    b = np.zeros((C_inv.shape[1], d))
    b[0, :] = X0
    b[1:, :] = Vhat[:N - 1, :]
    Xhat = np.dot(C_inv, b)
    return Xhat, Vhat


def SRLS(anchors, W, r2, printout=False):
    '''
    Squared range least squares (A)
    A.Beck, P.Stoica

    Args:
        anchors: anchor points
        r2: squared distances from anchors to point x.
    Returns:
        x: estimated position of point x.
    '''
    def y_hat(_lambda):
        lhs = ATA + _lambda * D
        rhs = (np.dot(A.T, b).reshape((-1, 1)) - _lambda * f).reshape((-1,))
        return np.linalg.solve(lhs, rhs)

    def phi(_lambda):
        yhat = y_hat(_lambda).reshape((-1, 1))
        return np.dot(yhat.T, np.dot(D, yhat)) + 2 * np.dot(f.T, yhat)

    from scipy import optimize
    from scipy.linalg import sqrtm
    # Set up optimization problem
    n = anchors.shape[0]
    d = anchors.shape[1]
    A = np.c_[-2 * anchors, np.ones((n, 1))]
    Sigma = np.diag(np.power(W, 0.5))
    A = Sigma.dot(A)
    ATA = np.dot(A.T, A)
    b = r2 - np.power(np.linalg.norm(anchors, axis=1), 2)
    b = Sigma.dot(b)
    D = np.zeros((d + 1, d + 1))
    D[:d, :d] = np.eye(d)
    if (printout):
        print('rank A:', A)
        print('ATA:', ATA)
        print('rank:', np.linalg.matrix_rank(A))
        print('ATA:', np.linalg.eigvals(ATA))
        print('D:', D)
        print('condition number:', np.linalg.cond(ATA))
    f = np.c_[np.zeros((1, d)), -0.5].T

    # Compute lower limit for lambda (s.t. AT*A+lambda*D psd)
    reg = 1
    sqrtm_ATA = sqrtm(ATA + reg * np.eye(ATA.shape[0]))
    B12 = np.linalg.inv(sqrtm_ATA)
    tmp = np.dot(B12, np.dot(D, B12))
    eig = np.linalg.eigvals(tmp)

    eps = 0.01
    if eig[0] == reg:
        print('eigenvalue equals reg.')
        I_orig = -1e5
    elif eig[0] > 1e-10:
        I_orig = -1.0 / eig[0] + eps
    else:
        print('smallest eigenvalue is zero')
        I_orig = -1e5
    inf = 1e5
    try:
        lambda_opt = optimize.bisect(phi, I_orig, inf)
    except:
        print('Bisect failed. Trying Newton...')
        lambda_opt = optimize.newton(phi, I_orig)
    if (printout):
        print('I', I)
        print('phi I', phi(I))
        print('phi inf', phi(inf))
        print('lambda opt', lambda_opt)
        print('phi opt', phi(lambda_opt))
        xvec = np.linspace(I, lambda_opt + abs(I), 100)
        y = list(map(lambda x: phi(x), xvec))
        y = np.array(y).reshape((-1,))
        plt.plot(xvec, y)
        plt.grid('on')
        plt.vlines(lambda_opt, y[0], y[-1])
        plt.hlines(0, xvec[0], xvec[-1])

    # Compute best estimate
    yhat = y_hat(lambda_opt)

    # By definition, y = [x^T, alpha] and by constraints, alpha=norm(x)^2
    assert_print(np.linalg.norm(yhat[:-1])**2 - yhat[-1], 1e-6)
    assert_print(phi(lambda_opt), 1e-6)
    lhs = np.dot(A.T, A) + lambda_opt * D
    rhs = (np.dot(A.T, b).reshape((-1, 1)) - lambda_opt * f).reshape((-1,))
    assert_all_print(np.dot(lhs, yhat) - rhs, 1e-6)
    eig = np.array(np.linalg.eigvals(ATA + lambda_opt * D))
    #assert (eig >= 0).all(), 'not all eigenvalues positive:{}'.format(eig)
    return yhat[:d]


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
        from plots_cti import plot_cost_function
        deltas = np.linspace(delta - 1.0, delta + 1.0, 100)
        fs = []
        for delta_x in deltas:
            X_delta = X_k.copy()
            X_delta[i, coord] += delta_x
            fs.append(f(X_delta, D, W))
        x_0 = X_k[i, coord]
        x_delta = X_k[i, coord] + delta[0]
        names = ['x', 'y', 'z']
        plot_cost_function(deltas, x_0, x_delta, fs,  names[coord])
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


def alternating_completion(edm, rank, mask, print_out=False, niter=50, tol=1e-6):
    from basics import low_rank_approximation
    N = edm.shape[0]
    edm_complete = edm.copy()
    edm_complete[mask == 0] = np.mean(edm)
    err = np.linalg.norm(edm_complete - edm)
    if print_out:
        print('iteration \t edm difference')
        print('0 \t {}'.format(err))
    errs = [err]
    for i in range(niter):
        # impose matrix rank
        edm_complete = low_rank_approximation(edm_complete, rank)

        # impose known entries
        edm_complete[mask] = edm[mask]

        # impose matrix structure
        edm_complete[range(N), range(N)] = 0.0
        edm_complete[edm_complete < 0] = 0.0
        edm_complete = 0.5 * (edm_complete + edm_complete.T)

        err = np.linalg.norm(edm_complete - edm)
        errs.append(err)
        if print_out:
            print('{} \t {}'.format(i + 1, err))
        if abs(errs[-2] - errs[-1]) < tol:
            break
    return edm_complete, errs


def reconstruct_emds(edm, Om, real_points):
    """
    Edge-MDS using distances and angles.
    """
    N = real_points.shape[0]
    d = real_points.shape[1]
    dm = dm_from_edm(edm)
    Xhat, __ = super_mds(Om, dm, real_points[0, :], N, d)
    Y, R, t, c = procrustes(real_points, Xhat, True)
    return Y


def reconstruct_mds(edm, real_points, completion='optspace', mask=None, method='geometric'):
    from point_configuration import dm_from_edm
    N = real_points.shape[0]
    d = real_points.shape[1]
    if mask is not None:
        if completion == 'optspace':
            from opt_space import opt_space
            edm_missing = np.multiply(edm, mask)
            X, S, Y, __ = opt_space(edm_missing, r=d, niter=500,
                                    tol=1e-6, print_out=False)
            edm = X.dot(S.dot(Y.T))
            edm[range(N), range(N)] = 0.0
        elif completion == 'alternate':
            edm, errs = alternating_completion(edm, d+2, mask)
        else:
            raise NameError('Unknown completion method {}'.format(completion))
    Xhat = MDS(edm, d, method, False).T
    Y, R, t, c = procrustes(real_points[:-1], Xhat, True)
    #Y, R, t, c = procrustes(real_points, Xhat, True)
    return Y


def reconstruct_srls(edm, points, plot=False, indices=[-1], W=None):
    Y = points.copy()
    for index in indices:
        anchors = np.delete(points, indices, axis=0)
        r2 = np.delete(edm[index, :], indices)
        if W is None:
            W = np.ones(edm.shape)
        w = np.delete(W[index, :], indices)
        # delete anchors where weight is zero to avoid ill-conditioning
        missing_anchors = np.where(w == 0.0)
        w = np.delete(w, missing_anchors)
        r2 = np.delete(r2, missing_anchors)
        anchors = np.delete(anchors, missing_anchors, axis=0)
        srls = SRLS(anchors, w, r2, plot)
        Y[index, :] = srls
    return Y


def reconstruct_acd(edm, W, X_0, real_points, print_out=False,):
    from point_configuration import create_from_points, PointConfiguration
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


if __name__ == "__main__":
    print('nothing happens when running this module.')
