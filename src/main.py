import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classifiers import NMC
from fun_utils import load_mnist, split_data
from sklearn.metrics import pairwise_distances
import time


def plot_digit(image, shape=(28, 28)):
    plt.imshow(np.reshape(image, newshape=shape), cmap='gray')

def plot_ten_digits(x, y=None):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plot_digit(x[i, :])
        if y is not None:
            plt.title('Label: ' + str(y[i]))

# measure test error
def ts_error(y_pred, yts):
    return (y_pred != yts).mean()

data_loader = load_mnist('data/mnist_data.csv')
clf = NMC()

x, y = data_loader

xtr, ytr, xts, yts = split_data(x, y)
clf.fit(xtr, ytr)
print(clf.centroids)

y_pred = clf.predict(xts)
ts_error = ts_error(y_pred, yts)

print(ts_error)