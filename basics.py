import numpy as np
import itertools
from math import pi, atan, atan2, sqrt, acos, cos, sin

def rmse(x, xhat):
    ''' Calcualte rmse between vector or matrix x and xhat '''
    sum_ = np.sum(np.power(x-xhat, 2))
    return sqrt(sum_/len(x))

def low_rank_approximation(A, r):
    ''' Returns approximation of A of rank r in least-squares sense.'''
    u, s, v = np.linalg.svd(A, full_matrices=False)
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar

def check_psd(G, print_out=False, print_rank=False):
    if not np.isclose(G, G.T).all():
        print('G not symmetric!')
    lamda, u = np.linalg.eig(G)
    lamda[np.abs(lamda)<1e-10] = 0
    if (print_rank):
        if np.linalg.matrix_rank(G,1e-10) != 2:
            print('rank:',np.linalg.matrix_rank(G,1e-10))
    psd = (lamda>=0).all()
    if not psd:
        if (print_out):
            print('G not psd: \n{}, eig:{}'.format(G, lamda))
    return min(lamda), sum(lamda>1e-10)

def eigendecomp(G, d):
    '''
    Computes sorted eigendecomposition of G.
    Params
        G:  Matrix 
        d:  rank
    Returns
        factor: vector of square root d of eigenvalues (biggest to smallest). 
        u: matrix with colums equal to the normalized eigenvectors corresponding to sorted eigenvalues.
    '''
    N = G.shape[0]
    lamda, u = np.linalg.eig(G)
    # test decomposition of G.
    G_hat = np.dot(np.dot(u,np.diag(lamda)),u.T)
    #assert np.linalg.norm(G_hat - G) < 1e-10, 'decomposition not exact: err {}'.format(np.linalg.norm(G_hat - G))

    factor = np.zeros((N,))
    # sort the eigenvalues in decreasing order
    indices = np.argsort(np.real(lamda))[::-1]
    lamda = np.real(lamda)
    lamda_sorted = lamda[indices]
    assert (lamda_sorted[:d] > -1e-10).all(), "{} not all positive!".format(lamda_sorted[:d])

    u = u[:,indices]
    factor[0:d] = np.sqrt(lamda_sorted[:d])
    return np.real(factor), np.real(u)

def change_angles(method, theta, tol=1e-10):
    ''' Function used by all angle conversion functions (from_x_to_x_pi(...))'''
    try:
        theta_new = np.zeros(theta.shape)
        for i,thet in enumerate(theta):
            try:
                # theta is vector
                theta_new[i] = method(thet,tol)
            except:
                # theta is matrix
                for j,th in enumerate(thet):
                    try:
                        theta_new[i,j] = method(th,tol)
                    except:
                        # theta is tensor
                        for k,t in enumerate(th):
                            theta_new[i,j,k] = method(t,tol)
        return theta_new
    except:
        return method(theta,tol)

def from_0_to_pi(theta):
    def from_0_to_pi_scalar(theta, tol):
        theta = from_0_to_2pi(theta)
        theta = min(theta, 2*pi-theta)
        assert theta >= 0 and theta <= pi, "{} not in [0, pi]".format(theta)
        return theta
    return change_angles(from_0_to_pi_scalar, theta)

def from_0_to_2pi(theta, tol=1e-10):
    def from_0_to_2pi_scalar(theta, tol):
        theta = theta % (2*pi)
        # eliminate numerical issues of % function
        if abs(theta-2*pi)<tol:
            theta=theta-2*pi
        if theta < 0:
            theta = 2*pi + theta
        assert theta >= 0 and theta <= 2*pi, "{} not in [0, 2pi]".format(theta)
        return theta
    return change_angles(from_0_to_2pi_scalar, theta, tol)

def get_absolute_angle(Pi, Pj):
    if (Pi == Pj).all():
        return 0
    y = Pj[1]-Pi[1]
    x = Pj[0]-Pi[0]
    theta_ij = atan2(y,x)
    return from_0_to_2pi(theta_ij)

def get_inner_angle(Pk, Pij):
    theta_ki = get_absolute_angle(Pk, Pij[0])
    theta_kj = get_absolute_angle(Pk, Pij[1])
    theta = abs(theta_ki - theta_kj)
    return from_0_to_pi(theta)
