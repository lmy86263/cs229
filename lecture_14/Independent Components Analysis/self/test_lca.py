import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
from lca import lca

rng = np.random.RandomState(42)

# test example from https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_vs_pca.html

# reference: https://blog.csdn.net/lizhe_dashuju/article/details/50263339

def test_data():
    S = rng.standard_t(1.5, size=(20000, 2))
    S[:, 0] *= 2.
    # Mix data
    # Mixing matrix
    A = np.array([[1, 1], [0, 2]])
    # Generate observations
    X = np.dot(S, A.T)
    return S, X


def pca(X):
    pca = PCA()
    S_pca_ = pca.fit(X).transform(X)
    return S_pca_, pca


def fastICA(X):
    ica = FastICA(random_state=rng)
    # Estimate the sources
    S_ica_ = ica.fit(X).transform(X)
    S_ica_ /= S_ica_.std(axis=0)
    return S_ica_, ica


def plot_samples(S, axis_list=None):
    plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,
                color='steelblue', alpha=0.5)
    if axis_list is not None:
        colors = ['orange', 'red']
        for color, axis in zip(colors, axis_list):
            axis /= axis.std()
            x_axis, y_axis = axis
            # Trick to get legend to work
            plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
            plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6,
                       color=color)

    plt.hlines(0, -3, 3)
    plt.vlines(0, -3, 3)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x')
    plt.ylabel('y')


if __name__ == '__main__':
    S, X = test_data()

    S_pca_, pca = pca(X)
    S_ica_, ica = fastICA(X)

    plt.figure()
    plt.subplot(3, 3, 1)
    plot_samples(S / S.std())
    plt.title('True Independent Sources')

    axis_list = [pca.components_.T, ica.mixing_]
    plt.subplot(3, 3, 2)
    plot_samples(X / np.std(X), axis_list=axis_list)
    legend = plt.legend(['PCA', 'ICA'], loc='upper right')
    legend.set_zorder(100)
    plt.title('Observations')

    plt.subplot(3, 3, 3)
    plot_samples(S_pca_ / np.std(S_pca_, axis=0))
    plt.title('PCA recovered signals')

    plt.subplot(3, 3, 4)
    plot_samples(S_ica_ / np.std(S_ica_))
    plt.title('ICA recovered signals')

    s_self = lca(X)
    plt.subplot(3, 3, 5)
    plot_samples(s_self / np.std(s_self))
    plt.title('ICA recovered self')

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
    plt.show()
