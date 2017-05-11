import numpy as np
import matplotlib.pyplot as plt
from math import pi, floor, cos, sin
from basics import rmse, eigendecomp

def MDS(D, dim, method='simple',theta=True):
    N = D.shape[0]
    def theta_from_eigendecomp(factor, u):
        theta_hat = np.dot(np.diag(factor[:]),u.T)
        theta_hat = theta_hat[0,:]
        return np.real(theta_hat).reshape((-1,))
    if method=='simple':
        d1 = D[0,:]
        G = -0.5*(D - d1*np.ones([1,N]).T - (np.ones([N,1])*d1).T)
        factor, u = eigendecomp(G,dim)
        if (theta):
            return theta_from_eigendecomp(factor, u)
        else:
            return np.dot(np.diag(factor[:]),u.T)[:dim,:]
    if method=='advanced':
        s1T = np.vstack([np.ones([1,N]),np.zeros([N-1,N])])
        G = -0.5 * np.dot(np.dot((np.identity(N) - s1T.T),D),(np.identity(N) - s1T))
        factor, u = eigendecomp(G,dim)
        if (theta):
            return theta_from_eigendecomp(factor, u)
        else:
            return np.dot(np.diag(factor[:]),u.T)[:dim,:]
    if method=='geometric':
        J = np.identity(N) - 1.0/float(N)*np.ones([N,N])
        G = -0.5 * np.dot(np.dot(J, D),J)
        factor, u = eigendecomp(G,dim)
        if (theta):
            return theta_from_eigendecomp(factor, u)
        else:
            return np.dot(np.diag(factor[:]),u.T)[:dim,:]

def classical_mds(D):
    return MDS(D,1,'geometric')

def procrustes(Y, X, scale=True):
    '''
    Given NA > d anchor nodes (Y in R^(NA x d)), return transformation
    of coordinates X optimally matching Y in least squares sense. (output of EDM algorithm)
    '''
    def centralize(X):
        n = X.shape[0]
        ones = np.ones((n,1))
        return X - np.multiply(1/n*np.dot(ones.T,X),ones)
    NA = Y.shape[0]
    N = X.shape[0]
    X_NA = X[:NA,:]
    ones = np.ones((NA,1))

    mux = 1/NA*np.dot(ones.T, X_NA)
    muy = 1/NA*np.dot(ones.T, Y)
    sigmax = 1/NA*np.linalg.norm(X_NA-mux)**2
    sigmaxy = 1/NA*np.dot((Y-muy).T,X_NA-mux)
    U,D,VT = np.linalg.svd(sigmaxy)
    #S = np.eye(D.shape[0])
    #this doesn't work and doesn't seem to be necessary! (why?)
    #assert abs(abs(np.linalg.det(U)*np.linalg.det(VT.T))-1) < 1e-10
    #if (np.linalg.det(U)*np.linalg.det(VT.T) < 0):
        #print('switching')
        #S[-1,-1] = -1.0
    #c = np.trace(np.dot(np.diag(D),S))/sigmax
    #R = np.dot(U, np.dot(S,VT))
    if (scale):
        c = np.trace(np.diag(D))/sigmax
    else:
        c = np.trace(np.diag(D))/sigmax
        assert abs(c-1) < 1e-10, 'scale not equal to 1: {}'.format(c)
        c = 1.0
    R = np.dot(U, VT)
    #t = np.dot(muy - c*np.dot(R, mux))
    t = muy.T - c*np.dot(R,mux.T)
    X_transformed2 = c*np.dot(R,X.T)  + t
    X_transformed = (c*np.dot(R,(X-mux).T)+muy.T).T
    return X_transformed, R, t, c

def super_mds(Om, dm, X0, N, d):
    from basics import eigendecomp
    factor, u = eigendecomp(Om, d)
    uhat = u[:,:d]
    lambdahat = np.diag(factor[:d])
    diag_dm = np.diag(dm)
    Vhat = np.dot(diag_dm, np.dot(uhat, lambdahat))
    C_inv = -np.eye(N)
    C_inv[0,0] = 1.0
    C_inv[:,0] = 1.0
    b = np.zeros((C_inv.shape[1],d))
    b[0,:] = X0
    b[1:,:] = Vhat[:N-1,:]
    Xhat = np.dot(C_inv,b)
    return Xhat, Vhat

def SRLS(anchors, r2, printout=False):
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
        lhs = np.dot(A.T, A) + _lambda*D
        rhs = (np.dot(A.T,b).reshape((-1,1)) - _lambda*f).reshape((-1,))
        return np.linalg.solve(lhs, rhs)

    def phi(_lambda):
        yhat = y_hat(_lambda).reshape((-1,1))
        return np.dot(yhat.T, np.dot(D, yhat)) + 2*np.dot(f.T,yhat)

    from scipy import optimize
    from scipy.linalg import sqrtm
    n = anchors.shape[0]
    d = anchors.shape[1]
    A = np.c_[-2*anchors, np.ones((n,1))]
    ATA = np.dot(A.T, A)
    b = r2-np.power(np.linalg.norm(anchors,axis=1),2)
    D = np.zeros((d+1,d+1))
    D[:d,:d] = np.eye(d)
    if (printout):
        print('rank A:',A)
        print('ATA:',ATA)
        print('rank:',np.linalg.matrix_rank(A))
        print('ATA:',np.linalg.eigvals(ATA))
        print('D:',D)
    f = np.c_[np.zeros((1,d)), -0.5].T

    # Compute lower limit for lambda (s.t. AT*A+lambda*D psd)
    B12 = np.linalg.inv(sqrtm(ATA))
    tmp = np.dot(B12, np.dot(D, B12))
    eig = np.linalg.eigvals(tmp)
    # TODO: what is wrong here? sometimes I_orig is not of the correct sign? 
    eps = 0.01
    I_orig = -1.0/eig[0] + eps
    inf = 1e5
    found = False
    counter = 0
    I = I_orig
    while not found:
        print('phi({})={}, phi({})={}'.format(I,phi(I), inf,phi(inf)))
        if phi(I) > 0 and phi(inf) < 0:
            found = True
        else:
            I -= I_orig
            counter += 1
        if counter > 100:
            print('did not find a good left limit!')
            break
    lambda_opt = optimize.bisect(phi, I, inf)

    if (printout):
        print('I',I)
        print('phi I',phi(I))
        print('phi inf',phi(inf))
        print('lambda opt',lambda_opt)
        print('phi opt',phi(lambda_opt))
        xvec = np.linspace(I,lambda_opt+abs(I),100)
        y = list(map(lambda x: phi(x), xvec))
        y = np.array(y).reshape((-1,))
        plt.plot(xvec,y)
        plt.grid('on')
        plt.vlines(lambda_opt, y[0], y[-1])
        plt.hlines(0, xvec[0], xvec[-1])

    # Compute best estimate
    yhat = y_hat(lambda_opt)

    assert (np.linalg.norm(yhat[:-1])**2 - yhat[-1]) < 1e-10
    assert abs(phi(lambda_opt)) < 1e-10
    lhs = np.dot(A.T, A) + lambda_opt*D
    rhs = (np.dot(A.T,b).reshape((-1,1)) - lambda_opt*f).reshape((-1,))
    assert np.allclose(np.dot(lhs,yhat), rhs)
    eig = np.array(np.linalg.eigvals(ATA + lambda_opt*D))
    assert (eig >= 0).all()

    return yhat[:d]

def reconstruct_mds(dm, points, plot=False, method='super',Om=''):
    N = points.shape[0]
    d = points.shape[1]
    if method == 'super':
        Xhat, __ = super_mds(Om, dm, points[0,:], N, d)
    else:
        triu_idx = np.triu_indices(n=N,m=N,k=1)
        # create edm from distances
        edm = np.zeros((N, N))
        edm[triu_idx[0],triu_idx[1]] = np.power(dm,2)
        edm = edm + edm.T
        Xhat = MDS(edm, d, method, False).T
    #Y, R, t, c = procrustes(points[:d+1],Xhat, True)
    Y, R, t, c = procrustes(points[:-1],Xhat, True)
    if (plot):
        from plots import plot_points
        plot_points(points,'original',[2,2])
        plot_points(Y,method,[2,2])
    return Y

def reconstruct_srls(dm, points, plot=False, index=-1):
    anchors = np.delete(points,index,axis=0)
    srls = SRLS(anchors, dm, plot)
    Y = points.copy()
    Y[index,:] = srls
    return Y
