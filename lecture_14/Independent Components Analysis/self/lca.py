import matplotlib.pyplot as plt

import numpy as np
import random

import copy


def sigmoid(s):
    """
        used to be cumulative distribution of observed data -> CDF(s)
    :param s:
    :return:
    """
    return 1 / (1 + np.exp(-1*s))


def sigmoid_derivative(s):
    """
        used to be probability density distribution of observed data -> PDF(s)
    :param s:
    :return:
    """
    return sigmoid(s) * (1 - sigmoid(s))


def laplace(s):
    """
        used to be probability density distribution of observed data -> PDF(s)
    :param s:
    :return:
    """
    return 0.5 * np.exp(-1*np.absolute(s))


def lca(x, batch_size=7, epochs=20):
    """
    :param x: observed data
    :param batch_size: random training samples for SGD
    :param epochs: training times
    :return:
    """
    # learning process, stochastic gradient descent
    x_local = copy.deepcopy(x)
    alpha = 0.1
    dimension_s = x_local.shape[1]
    # initial parameters for w
    w = np.random.random((dimension_s, dimension_s))

    for i in range(epochs):
        random.shuffle(x_local)
        x_train = x_local[:batch_size, :]

        # gradient = np.zeros((dimension_s, dimension_s))
        for item in x_train:
            part1 = 1 - 2 * sigmoid(np.dot(w, item.T))
            part2 = np.dot(part1[:, None], item[None, :])
            gradient = part2 + np.linalg.inv(w.T)
            # gradient += g
            w = w - alpha * gradient

    s = np.dot(x, w)
    return s


if __name__ == '__main__':
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    y_prime = sigmoid_derivative(x)
    z = laplace(x)
    plt.plot(x, y, color='red')
    plt.plot(x, y_prime, color='blue')
    plt.plot(x, z, color='green')
    plt.show()

