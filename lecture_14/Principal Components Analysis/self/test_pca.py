import random
import pickle
import os

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from pca import pca, pca_with_svd


def test_data():
    a = np.array([[1, 2, 3, 1, 1.56, 100.1],
                  [4, 5, 6, 2, 2.45, 50.2],
                  [7, 8, 9, 3, 6.23, 34.2]])
    return a


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
