from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    '''
     return x_tr, y_tr, x_ts, y_ts
    '''
    n_samples = y.size
    n_tr = int(n_samples * tr_fraction)
    idx = np.array(range(0, n_samples))
    np.random.shuffle(idx)

    tr_idx = idx[:n_tr]
    ts_idx = idx[n_tr:]

    x_tr = x[tr_idx, :]
    y_tr = y[tr_idx]

    x_ts = x[ts_idx, :]
    y_ts = y[ts_idx]

    return x_tr, y_tr, x_ts, y_ts

def load_mnist(csv_filename: str):
    """
    Load the MNIST dataset from a csv file

    Parameters
    ----------
    csv_filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    return load_data(csv_filename)

def plot_ten_images(X, h=28, w=28):
    """
    Plots ten images in a single figure

    Parameters
    ----------
    X: the images given as a matrix of size: (10, h*w)
    h: the image height
    w: the image width

    Returns
    -------
    None.
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(X[i].reshape((h, w)), cmap=plt.cm.gray)
        ax.axis('off')