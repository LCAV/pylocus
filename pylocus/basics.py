import numpy as np
import itertools
from math import pi, atan, atan2, sqrt, acos, cos, sin


def mse(x, xhat):
    """ Calcualte mse between vector or matrix x and xhat """
    buf_ = x - xhat
    np.square(buf_, out=buf_)  # square in-place
    sum_ = np.sum(buf_)
    sum_ /= x.size  # divide in-place
    return sum_


def rmse(x, xhat):
    """ Calcualte rmse between vector or matrix x and xhat """
    ms_error = mse(x, xhat)
    return sqrt(ms_error)


def norm(x, xhat):
    buf_ = x - xhat
    np.square(buf_, out=buf_)  # square in-place
    sum_ = np.sum(buf_)
    return sqrt(sum_)


def low_rank_approximation(A, r):
    """ Returns approximation of A of rank r in least-squares sense."""
    try:
        u, s, v = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError as e:
        print('Matrix:', A)
        print('Matrix rank:', np.linalg.matrix_rank(A))
        raise

    Ar = np.zeros((len(u), len(v)), dtype=u.dtype)
    buf_ = np.empty_like(Ar)
    sc_vec_ = np.empty((v.shape[1],), dtype=v.dtype)
    for i in range(r):
        np.multiply(v[i], s[i], out=sc_vec_)
        np.outer(u[:, i], sc_vec_, out=buf_)
        Ar += buf_
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
    #G_hat = np.dot(np.dot(u, np.diag(lamda)), u.T)
    #assert np.linalg.norm(G_hat - G) < 1e-10, 'decomposition not exact: err {}'.format(np.linalg.norm(G_hat - G))

    # sort the eigenvalues in decreasing order
    lamda = np.real(lamda)
    indices = np.argsort(lamda)[::-1]
    lamda_sorted = lamda[indices]
    assert (lamda_sorted[
            :d] > -1e-10).all(), "{} not all positive!".format(lamda_sorted[:d])

    # Set the small negative values of 
    # lamda to zero.
    lamda_sorted[lamda_sorted < 0] = 0

    u = u[:, indices]
    factor = np.empty((N,), dtype=lamda.dtype)
    np.sqrt(lamda_sorted[:d], out=factor[0:d])
    factor[d:] = 0.0
    return np.real(factor), np.real(u)


def assert_print(this_should_be_less_than, this=1e-10):
    assert abs(this_should_be_less_than) < this, "abs({}) not smaller than {}".format(
        this_should_be_less_than, this)


def assert_all_print(this_should_be_less_than, this=1e-10):
    assert (np.abs(this_should_be_less_than) < this).all(
    ), "abs({}) not smaller than {}".format(this_should_be_less_than, this)


def divide_where_nonzero(divide_this, by_this):
    result = np.empty_like(divide_this)
    zero_mask = (by_this == 0)
    if zero_mask.size:
        result[zero_mask] = 0.0
        nonzero_mask = zero_mask                         # creates a view
        # overwrites memory of zero_mask
        np.logical_not(zero_mask, out=nonzero_mask)
        result[nonzero_mask] = divide_this[nonzero_mask] / \
            by_this[nonzero_mask]
    else:
        # more efficient, in-place operation
        np.divide(divide_this, by_this, out=result)
    return result


def get_rotation_matrix(thetas):
    theta_x, theta_y, theta_z = thetas
    # N.B.:
    # because Rx, Ry, Rz are all rotations in YZ plane, composition of these rotations
    # is equivalent to one rotation through angle theta_x + theta_y + theta_z
    th_xyz = theta_x + theta_y + theta_z
    cxyz = np.cos(th_xyz)
    sxyz = np.sin(th_xyz)
    return np.array([[1, 0, 0], [0, cxyz, sxyz], [0, -sxyz, cxyz]])


def get_edm(X):
    N = X.shape[0]
    rows, cols = np.indices((N, N))
    buf_ = X[rows, :] - X[cols, :]
    np.square(buf_, out=buf_)
    edm = np.sum(buf_, axis=2)
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
    tmp_ = A.dot(x)
    tmp_ -= b
    x_hat = A_pseudoinv.dot(tmp_)
    np.subtract(x, x_hat, out=x_hat)

    cost = mse(x_hat, x)
    A.dot(x_hat, out=tmp_)
    constraints_error = mse(tmp_, b)
    return x_hat, cost, constraints_error


def vector_from_matrix(matrix):
    N = matrix.shape[0]
    triu_idx = np.triu_indices(n=N, m=N, k=1)
    return matrix[triu_idx]


def matrix_from_vector(vector, N):
    triu_idx = np.triu_indices(n=N, m=N, k=1)
    matrix = np.zeros((N, N), dtype=vector.dtype)
    matrix[triu_idx[0], triu_idx[1]] = vector
    return matrix
