#!/usr/bin/env python
# module MDS
import numpy as np
from .basics import eigendecomp


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


def superMDS(X_0, N, d, **kwargs):
    Om = kwargs.get('Om',None)
    dm = kwargs.get('dm',None)
    if Om is not None and dm is not None:
        KE = kwargs.get('KE',None)
        if KE is not None:
            print('superMDS: KE and Om, dm given. Continuing with Om, dm')
        factor, u = eigendecomp(Om, d)
        uhat = u[:, :d]
        lambdahat = np.diag(factor[:d])
        diag_dm = np.diag(dm)
        Vhat = np.dot(diag_dm, np.dot(uhat, lambdahat))
    elif Om is None or dm is None:
        KE = kwargs.get('KE',None)
        if KE is None:
            raise NameError('Either KE or Om and dm have to be given.')
        factor, u = eigendecomp(KE, d)
        lambda_ = np.diag(factor)
        Vhat = np.dot(u,lambda_)[:,:d]

    C_inv = -np.eye(N)
    C_inv[0, 0] = 1.0
    C_inv[:, 0] = 1.0
    b = np.zeros((C_inv.shape[1], d))
    b[0, :] = X_0
    b[1:, :] = Vhat[:N - 1, :]
    Xhat = np.dot(C_inv, b)
    return Xhat, Vhat


if __name__ == "__main__":
    print('nothing happens when running this module. It is only a container of functions.')
