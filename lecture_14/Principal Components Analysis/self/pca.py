from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def test_data():
    a = np.array([[1, 2, 3, 1, 1.56, 100.1],
                  [4, 5, 6, 2, 2.45, 50.2],
                  [7, 8, 9, 3, 6.23, 34.2]])
    return a


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


def word_vectors():
    is_words_exist = os.path.exists('word.pickle')
    is_vector_exist = os.path.exists('word_vectors.pickle')

    if is_words_exist and is_vector_exist:
        f1 = open('words.pickle', 'rb')
        data = pickle.load(f1)
        f2 = open('word_vectors.pickle', 'rb')
        word_vectors = pickle.load(f2)
    else:
        glove_file = datapath('/Users/lmy86263/Corpora/GloVe/glove.6B/glove.6B.100d.txt')
        word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
        glove2word2vec(glove_file, word2vec_glove_file)

        model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
        words = [word for word in model.vocab]
        random.shuffle(words)
        data = [word for i, word in enumerate(words) if i < 10]
        f1 = open('words.pickle', 'wb')
        pickle.dump(data, f1)
        word_vectors = np.array([model[w] for w in data])
        f2 = open('word_vectors.pickle', 'wb')
        pickle.dump(word_vectors, f2)
    return data, word_vectors


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


if __name__ == '__main__':
    # data, vectors = word_vectors()
    vectors = test_data()
    n_samples = vectors.shape[0]
    reduced_dimension = 2

    reduced_x = PCA(n_components=reduced_dimension).fit_transform(vectors)
    reduced_x_self = pca(vectors, n_dimensions=reduced_dimension)
    reduced_x_svd_self = pca_with_svd(vectors, n_dimensions=reduced_dimension)

    # for a matrix, eigenvectors can be a family, they are linear dependent,
    # you can just choose any one of them, make sure they are orthogonal and can
    # represent the direction of maximum variance.
    for i, word in enumerate(range(n_samples)):
        plt.scatter(reduced_x[i, 0], reduced_x[i, 1], marker='x', color='red', alpha=0.7)
        plt.text(reduced_x[i, 0], reduced_x[i, 1]+1,
                 '(%f, %f)' % (reduced_x[i, 0], reduced_x[i, 1]), fontsize=9)

    for i, word in enumerate(range(n_samples)):
        plt.scatter(reduced_x_self[i, 0], -reduced_x_self[i, 1], marker='o', color='blue', alpha=0.5)
        plt.text(reduced_x_self[i, 0], -reduced_x_self[i, 1]+0.5,
                 '(%f, %f)' % (reduced_x_self[i, 0], -reduced_x_self[i, 1]), fontsize=8)

    for i, word in enumerate(range(n_samples)):
        plt.scatter(-reduced_x_svd_self[i, 0], -reduced_x_svd_self[i, 1], marker='v', color='yellow', alpha=0.3)
        plt.text(-reduced_x_svd_self[i, 0], -reduced_x_svd_self[i, 1],
                 '(%f, %f)' % (-reduced_x_svd_self[i, 0], -reduced_x_svd_self[i, 1]), fontsize=7)
    x_ticks = np.arange(-35, 70, 10)
    y_ticks = np.arange(-3, 5, 1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.show()
