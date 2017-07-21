#!/usr/bin/env python
# module LATERATION

def SRLS(anchors, W, r2, print_out=False):
    '''Squared range least squares (A)

    Algorithm written by A.Beck, P.Stoica in "Approximate and Exact solutions of Source Localization Problems".

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
    if (print_out):
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
    if (print_out):
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
