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
    return sqrt(ms_error)


def norm(x, xhat):
    return sqrt(np.sum(np.power(x - xhat, 2)))


def low_rank_approximation(A, r):
    """ Returns approximation of A of rank r in least-squares sense."""
    try:
        u, s, v = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError as e:
        print('Matrix:', A)
        print('Matrix rank:', np.linalg.matrix_rank(A))
        raise

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

    # sort the eigenvalues in decreasing order
    indices = np.argsort(np.real(lamda))[::-1]
    lamda = np.real(lamda)
    lamda_sorted = lamda[indices]
    assert (lamda_sorted[
            :d] > -1e-10).all(), "{} not all positive!".format(lamda_sorted[:d])

    u = u[:, indices]
    factor = np.zeros((N,))
    factor[0:d] = np.sqrt(lamda_sorted[:d])
    return np.real(factor), np.real(u)


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


def vector_from_matrix(matrix):
    N = matrix.shape[0]
    triu_idx = np.triu_indices(n=N, m=N, k=1)
    return matrix[triu_idx]


def matrix_from_vector(vector, N):
    triu_idx = np.triu_indices(n=N, m=N, k=1)
    matrix = np.zeros((N, N))
    matrix[triu_idx[0], triu_idx[1]] = vector
    return matrix


if __name__ == "__main__":
    print('nothing happens when running this module.')
