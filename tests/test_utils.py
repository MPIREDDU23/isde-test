from src.fun_utils import load_mnist, split_data, plot_ten_images
import unittest
import numpy as np

mnist_path = 'data/mnist_data.csv'

class TestUtils(unittest.TestCase):
  def test_load_mnist(self):
    X, y = load_mnist(mnist_path)
    self.assertEqual(X.shape[0], y.shape[0], "Number of samples in X and y should be equal")
    self.assertEqual(X.shape[1], 784, "Each MNIST image should have 784 features (28x28 pixels)")

  def test_split_data(self):
    X, y = load_mnist(mnist_path)
    X_tr, y_tr, X_ts, y_ts = split_data(X, y, tr_fraction=0.7)
    total_samples = y.shape[0]
    expected_tr_samples = int(total_samples * 0.7)
    expected_ts_samples = total_samples - expected_tr_samples

    self.assertEqual(X_tr.shape[0], expected_tr_samples, "Training set size is incorrect")
    self.assertEqual(y_tr.shape[0], expected_tr_samples, "Training labels size is incorrect")
    self.assertEqual(X_ts.shape[0], expected_ts_samples, "Test set size is incorrect")
    self.assertEqual(y_ts.shape[0], expected_ts_samples, "Test labels size is incorrect")

  def test_plot_ten_images(self):
    X, y = load_mnist(mnist_path)
    plot_ten_images(X, h=28, w=28)