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


def procrustes(Y, X, scale=True):
    '''
    Given NA > d anchor nodes (Y in R^(NA x d)), return transformation
    of coordinates X optimally matching Y in least squares sense. (output of EDM algorithm)
    '''
    def centralize(X):
        n = X.shape[0]
        ones = np.ones((n, 1))
        return X - np.multiply(1 / n * np.dot(ones.T, X), ones)
    NA = Y.shape[0]
    N = X.shape[0]
    X_NA = X[:NA, :]
    ones = np.ones((NA, 1))

    mux = 1 / NA * np.dot(ones.T, X_NA)
    muy = 1 / NA * np.dot(ones.T, Y)
    sigmax = 1 / NA * np.linalg.norm(X_NA - mux)**2
    sigmaxy = 1 / NA * np.dot((Y - muy).T, X_NA - mux)
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


def SRLS(anchors, weights, r2, printout=False):
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
    Sigma = np.diag(np.power(weights, 0.5))
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
        print('condition number:',np.linalg.cond(ATA))
    f = np.c_[np.zeros((1, d)), -0.5].T

    # Compute lower limit for lambda (s.t. AT*A+lambda*D psd)
    reg = 1
    sqrtm_ATA = sqrtm(ATA + reg*np.eye(ATA.shape[0]))
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
        for j in range(N):
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


def reconstruct_mds(edm, points, plot=False, method='super', Om=''):
    from point_configuration import dm_from_edm
    N = points.shape[0]
    d = points.shape[1]
    if method == 'super':
        dm = dm_from_edm(edm)
        Xhat, __ = super_mds(Om, dm, points[0, :], N, d)
    else:
        Xhat = MDS(edm, d, method, False).T
    Y, R, t, c = procrustes(points[:-1], Xhat, True)
    return Y


def reconstruct_srls(edm, points, plot=False, index=-1, weights=None):
    anchors = np.delete(points, index, axis=0)

    r2 = np.delete(edm[index,:],index)
    if weights is None:
        weights = np.ones(edm.shape)
    w = np.delete(weights[index,:],index)

    # delete anchors where weight is zero to avoid ill-conditioning
    missing_anchors = np.where(w==0.0)
    w = np.delete(w, missing_anchors)
    r2 = np.delete(r2, missing_anchors)
    other = np.delete(range(edm.shape[0]), missing_anchors)
    anchors = np.delete(anchors, missing_anchors, axis=0)

    srls = SRLS(anchors, w, r2, plot)
    Y = points.copy()
    Y[index, :] = srls
    return Y


def reconstruct_weighted(edm, weights, X_0, X_hat, points, print_out=False,):
    from point_configuration import create_from_points, PointConfiguration
    X_k = X_0.copy()
    N = X_k.shape[0]
    d = X_k.shape[1]

    # create reference object
    preal = create_from_points(points, PointConfiguration)
    # create optimization object
    cd = create_from_points(X_k, PointConfiguration)

    if print_out:
        print('=======initialization=======')
        print('---mds---- edm    ', np.linalg.norm(cd.edm - edm))
        print('---mds---- points ', np.linalg.norm(X_k - X_hat))
        print('---real--- edm    ', np.linalg.norm(cd.edm - preal.edm))
        print('---real--- points ', np.linalg.norm(X_k - preal.points))
        print('cost function:', f(X_k, edm, weights))
    fs = []
    err_edms = []
    err_points = []
    done = False
    for counter in range(20):
        # sweep
        for i in range(N):
            for coord in range(d):
                delt = get_step_size(i, coord, X_k, edm, weights)
                if len(delt) > 1:
                    ftest = []
                    for de in delt:
                        X_ktest = X_k.copy()
                        X_ktest[i, coord] += de
                        ftest.append(f(X_ktest, edm, weights))
                    delt = delt[ftest==min(ftest)]
                X_k[i, coord] += delt
                f_this = f(X_k, edm, weights)
                fs.append(f_this)
                cd.points = X_k
                cd.init()
                err_edms.append(np.linalg.norm(cd.edm - edm))
                err_points.append(np.linalg.norm(X_k - preal.points))
                if len(fs) > 2 and i==0 and coord == 0:
                    if abs(fs[-1] - fs[-2]) < 1e-4:
                        print('weighted mds: converged after {} sweeps.'.format(counter+1))
                        return X_k, fs, err_edms, err_points
                    else:
                        print('err after sweep {}: {}'.format(counter+1,abs(fs[-1] - fs[-2])))
        if (print_out):
            print('======= step {} ======='.format(counter))
            print('---mds---- edm    ', np.linalg.norm(cd.edm - edm))
            print('---mds---- points ', np.linalg.norm(X_k - X_hat))
            print('---real--- edm    ', np.linalg.norm(cd.edm - preal.edm))
            print('---real--- points ', np.linalg.norm(X_k - preal.points))
            print('cost function:', f(X_k, edm, weights))
    print('weighted mds: did not converge after {} sweeps'.format(counter+1))
    return X_k, fs, err_edms, err_points


if __name__ == "__main__":
    print('nothing happens when running this module.')
