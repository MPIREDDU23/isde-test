import unittest
import numpy as np
from src.data_perturb.data_perturb_gaussian import CDataPerturbGaussian
from src.data_perturb.data_perturb_random import CDataPerturbRandom
from src.classifiers.nmc import NMC
from src.fun_utils import load_mnist, split_data

mnist_path = 'data/mnist_data.csv'

class TestBaseDataPerturbation(unittest.TestCase):
  def setUp(self):
    # Create a simple dataset for testing
    self.data = load_mnist(mnist_path)
    X, y = self.data
    self.X_train, self.y_train, self.X_test, self.y_test = split_data(X, y, tr_fraction=0.6)

class TestDataPerturbGaussian(TestBaseDataPerturbation):
  def test_gaussian_perturbation(self):
    perturb = CDataPerturbGaussian()
    X_perturbed = perturb.perturb_dataset(self.X_train)
    self.assertFalse(np.array_equal(X_perturbed, self.X_train), "Perturbed data should differ from original data")

class TestDataPerturbRandom(TestBaseDataPerturbation):
  def test_random_perturbation(self):
    perturb = CDataPerturbRandom()
    X_perturbed = perturb.perturb_dataset(self.X_train)
    self.assertFalse(np.array_equal(X_perturbed, self.X_train), "Perturbed data should differ from original data")

class TestAbstractDataPerturbation(TestBaseDataPerturbation):
  def test_abstract_perturbation(self):
    from src.data_perturb import CDataPerturb
    with self.assertRaises(TypeError):
      CDataPerturb()

    class DummyPerturb(CDataPerturb):
      def data_perturbation(self, data):
        super().data_perturbation(data)

    dummy = DummyPerturb()
    with self.assertRaises(NotImplementedError):
      dummy.data_perturbation(self.X_train)
