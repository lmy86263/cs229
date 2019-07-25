from sklearn.preprocessing import scale
import numpy as np


def preprocess(X):
    """
        R(N*M)
    :param X:
    :return:
    """
    X_average = X.mean(axis=0)
    X = X - X_average
    # sklearn does'nt make the variance to 1, cs229 suggest to do that.
    # that a difference
    # std_sigma = X.std(axis=0)
    # X = X / std_sigma

    X_scale = scale(X, axis=0)
    # assert (X == X_scale).all(), 'preprocess error'

    return X


def pca_with_svd(X, n_dimensions=2):
    X = preprocess(X)
    u, d, v_t = np.linalg.svd(X)

    result_vector = v_t[:n_dimensions, :]
    reduced_x_self = np.dot(X, result_vector.T)
    return reduced_x_self


def pca(X, n_dimensions=2, algorithm='default'):
    """

    :param n_dimensions: the dimension of subspace of original space
    :param algorithm: can use default and svd method
    :return:
    """
    X = preprocess(X)

    if algorithm == 'svd':
        return pca_with_svd(X, n_dimensions)

    dimensions = X.shape[1]
    n_samples = X.shape[0]

    Sigma = np.zeros((dimensions, dimensions))
    for index in range(n_samples):

        co_occurrence = np.dot(X[index, :][:, None], X[index, :][None, :])
        Sigma += np.cov(co_occurrence)

    Sigma = Sigma/n_samples

    # eigenvalues can be duplicated
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)

    eig_vals_sorted = np.sort(eigenvalues)
    eig_vecs_sorted = eigenvectors[:, eigenvalues.argsort()]

    result_eigen_vectors = eig_vecs_sorted[:, -n_dimensions:]

    reduced_x_self = np.flip(np.dot(X, result_eigen_vectors))
    return reduced_x_self