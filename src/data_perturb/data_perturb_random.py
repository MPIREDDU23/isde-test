from data_perturb.data_perturb import CDataPerturb
import numpy as np

class CDataPerturbRandom(CDataPerturb):
  '''
  - CDataPerturbRandom -
  class that implements random data perturbation

  the constructor takes as input:
  - K: number of values to be randomly changed in the input data vector (default 100)
  - min_value: minimum value for the random perturbation (default 0)
  - max_value: maximum value for the random perturbation (default 255)

  for each parameter, setter and getter methods are defined

  the method data_perturbation(data) implements the random perturbation by randomly changing K values in the input data vector, selecting such values uniformly in the range [min_value, max_value]
  '''
  def __init__(self, K=100, min_value=0, max_value=255):
    self.K = K
    self.min_value = min_value
    self.max_value = max_value

  # define setter and getter methods for K, min_value, max_value
  @property
  def K(self):
    return self._K
  @K.setter
  def K(self, value):
    self._K = value

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

    return a perturbed version of the input data by adding random noise

    randomly changs K values in the input data vector x, selecting such values uniformly in the range [min_value, max_value]
    '''
    perturbed_data = data.copy()
    data_length = len(data)
    indices = np.random.choice(data_length, self.K, replace=False)
    for idx in indices:
      #
      perturbed_data[idx] = np.random.uniform(self.min_value, self.max_value)
    return perturbed_data