from data_perturb import CDataPerturb
import numpy as np

class CDataPerturbGaussian(CDataPerturb):
  '''
  - CDataPerturbGaussian -
  Perturbs all values in the input data vector x by adding Gaussian noise.

  The gaussian noise must have zero meand and standard deviation parametrized by sigma

  Hint: use sigma * np.random.randn() to rescale the values sampled from the standard normal with zero mean and unit variance.

  if the values in the perturbed image are below min_value or above max_value, they should be set to min_value or max_value, respectively.

  The constructor will taker min_value, max_value and sigma as input parameters, having default values of 0, 255 and 100.0.

  Setter/getter methods should be defined for all parameters.
  '''
  def __init__(self, sigma=100.0, min_value=0, max_value=255):
    self.sigma = sigma
    self.min_value = min_value
    self.max_value = max_value

  # define setter and getter methods for sigma, min_value, max_value
  @property
  def sigma(self):
    return self._sigma
  @sigma.setter
  def sigma(self, value):
    self._sigma = value

  @property
  def min_value(self):
    return self._min_value
  @min_value.setter
  def min_value(self, value):
    self._min_value = value

  @property
  def max_value(self):
    return self._max_value
  @max_value.setter
  def max_value(self, value):
    self._max_value = value

  # implement the data_perturbation method
  def data_perturbation(self, data):
    '''
    - data_perturbation -
    input: data (flat vector of float)

    return a perturbed version of the input data by adding Gaussian noise

    all values in the input data vector x are perturbed by adding Gaussian noise with zero mean and standard deviation sigma
    if the values in the perturbed image are below min_value or above max_value, they should be set to min_value or max_value, respectively.
    '''
    perturbed_data = data.copy()
    for i in range(len(data)):
      noise = self.sigma * np.random.randn()
      perturbed_value = perturbed_data[i] + noise
      # clamp the value to be within [min_value, max_value]
      perturbed_value = np.clip(perturbed_value, self.min_value, self.max_value)
      perturbed_data[i] = perturbed_value
    return perturbed_data