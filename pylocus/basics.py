import numpy as np
import itertools
from math import pi, atan, atan2, sqrt, acos, cos, sin

def mse(x, xhat):
    """ Calcualte mse between vector or matrix x and xhat """
    sum_ = np.sum(np.power(x - xhat, 2))
    return sum_ / x.size


def rmse(x, xhat):
    """ Calcualte rmse between vector or matrix x and xhat """
    ms_error = mse(x, xhat)
    if ms_error:
        return sqrt(ms_error)
    else:
        return 0


def low_rank_approximation(A, r):
    """ Returns approximation of A of rank r in least-squares sense."""
    u, s, v = np.linalg.svd(A, full_matrices=False)
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar


def check_psd(G, print_out=False, print_rank=False):
    if not np.isclose(G, G.T).all():
        print('G not symmetric!')
    lamda, u = np.linalg.eig(G)
    lamda[np.abs(lamda) < 1e-10] = 0
    if (print_rank):
        if np.linalg.matrix_rank(G, 1e-10) != 2:
            print('rank:', np.linalg.matrix_rank(G, 1e-10))
    psd = (lamda >= 0).all()
    if not psd:
        if (print_out):
            print('G not psd: \n{}, eig:{}'.format(G, lamda))
    return min(lamda), sum(lamda > 1e-10)


def eigendecomp(G, d):
    """
    Computes sorted eigendecomposition of G.

    :param G:  Matrix 
    :param d:  rank

    :return factor: vector of square root d of eigenvalues (biggest to smallest). 
    :return u: matrix with colums equal to the normalized eigenvectors corresponding to sorted eigenvalues.
    """
    N = G.shape[0]
    lamda, u = np.linalg.eig(G)
    # test decomposition of G.
    G_hat = np.dot(np.dot(u, np.diag(lamda)), u.T)
    #assert np.linalg.norm(G_hat - G) < 1e-10, 'decomposition not exact: err {}'.format(np.linalg.norm(G_hat - G))

    factor = np.zeros((N,))
    # sort the eigenvalues in decreasing order
    indices = np.argsort(np.real(lamda))[::-1]
    lamda = np.real(lamda)
    lamda_sorted = lamda[indices]
    assert (lamda_sorted[
            :d] > -1e-10).all(), "{} not all positive!".format(lamda_sorted[:d])

    u = u[:, indices]
    factor[0:d] = np.sqrt(lamda_sorted[:d])
    return np.real(factor), np.real(u)


def change_angles(method, theta, tol=1e-10):
    """ Function used by all angle conversion functions (from_x_to_x_pi(...))"""
    try:
        theta_new = np.zeros(theta.shape)
        for i, thet in enumerate(theta):
            try:
                # theta is vector
                theta_new[i] = method(thet, tol)
            except:
                # theta is matrix
                for j, th in enumerate(thet):
                    try:
                        theta_new[i, j] = method(th, tol)
                    except:
                        # theta is tensor
                        for k, t in enumerate(th):
                            theta_new[i, j, k] = method(t, tol)
        return theta_new
    except:
        return method(theta, tol)


def from_0_to_pi(theta):
    def from_0_to_pi_scalar(theta, tol):
        theta = from_0_to_2pi(theta)
        theta = min(theta, 2 * pi - theta)
        assert theta >= 0 and theta <= pi, "{} not in [0, pi]".format(theta)
        return theta
    return change_angles(from_0_to_pi_scalar, theta)


def from_0_to_2pi(theta, tol=1e-10):
    def from_0_to_2pi_scalar(theta, tol):
        theta = theta % (2 * pi)
        # eliminate numerical issues of % function
        if abs(theta - 2 * pi) < tol:
            theta = theta - 2 * pi
        if theta < 0:
            theta = 2 * pi + theta
        assert theta >= 0 and theta <= 2 * \
            pi, "{} not in [0, 2pi]".format(theta)
        return theta
    return change_angles(from_0_to_2pi_scalar, theta, tol)


def get_absolute_angle(Pi, Pj):
    if (Pi == Pj).all():
        return 0
    y = Pj[1] - Pi[1]
    x = Pj[0] - Pi[0]
    theta_ij = atan2(y, x)
    return from_0_to_2pi(theta_ij)


def get_inner_angle(Pk, Pij):
    theta_ki = get_absolute_angle(Pk, Pij[0])
    theta_kj = get_absolute_angle(Pk, Pij[1])
    theta = abs(theta_ki - theta_kj)
    return from_0_to_pi(theta)


def assert_print(this_should_be_less_than, this=1e-10):
    assert abs(this_should_be_less_than) < this, "abs({}) not smaller than {}".format(
        this_should_be_less_than, this)


def assert_all_print(this_should_be_less_than, this=1e-10):
    assert (np.abs(this_should_be_less_than) < this).all(
    ), "abs({}) not smaller than {}".format(this_should_be_less_than, this)


def divide_where_nonzero(divide_this, by_this):
    result = np.zeros(divide_this.shape)
    result[by_this != 0] = divide_this[by_this != 0] / by_this[by_this != 0]
    return result


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


def get_rotation_matrix(thetas):
    theta_x, theta_y, theta_z = thetas
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    Rx = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    Ry = np.array([[1, 0, 0], [0, cy, sy], [0, -sy, cy]])
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    Rz = np.array([[1, 0, 0], [0, cz, sz], [0, -sz, cz]])
    return Rx.dot(Ry.dot(Rz))


def get_edm(X):
    N = X.shape[0]
    rows, cols = np.indices((N, N))
    edm = np.sum((X[rows, :] - X[cols, :])**2, axis=2)
    return edm


def pseudo_inverse(A):
    inv = np.linalg.inv(A.dot(A.T))
    return A.T.dot(inv)


def projection(x, A, b):
    """ Returns the vector xhat closest to x in 2-norm, satisfying A.xhat =b.

    :param x: vector
    :param A, b: matrix and array characterizing the constraints on x (A.x = b)

    :return x_hat:  optimum angle vector, minimizing cost.
    :return cost: least square error of xhat, x
    :return constraints_error: mse of constraint.
    :rtype: (numpy.ndarray, float, float)
    """
    A_pseudoinv = pseudo_inverse(A)
    x_hat = x - A_pseudoinv.dot(A.dot(x) - b)

    cost = mse(x_hat, x)
    constraints_error = mse(A.dot(x_hat), b)
    return x_hat, cost, constraints_error
