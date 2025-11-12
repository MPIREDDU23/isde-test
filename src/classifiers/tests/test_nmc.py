from classifiers.nmc import NMC
import unittest
import numpy as np
from src.fun_utils import load_mnist, split_data

mnist_path = 'data/mnist_data.csv'

class TestNMCClassifier(unittest.TestCase):
  def setUp(self):
    # Create a simple dataset for testing
    np.random.seed(0)
    self.data = load_mnist(mnist_path)
    X, y = self.data
    self.X_train, self.y_train, self.X_test, self.y_test = split_data(X, y, tr_fraction=0.6)

  def test_nmc_classifier(self):
    classifier = NMC()
    classifier.fit(self.X_train, self.y_train)
    y_pred = classifier.predict(self.X_test)
    accuracy = np.mean(y_pred == self.y_test)
    self.assertGreater(accuracy, 0.5, "NMC classifier should achieve accuracy greater than 50%")

  def test_nmc_without_fit(self):
    classifier = NMC()
    with self.assertRaises(ValueError):
      classifier.predict(self.X_test)