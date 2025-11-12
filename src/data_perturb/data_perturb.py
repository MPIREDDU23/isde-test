from abc import ABC, abstractmethod

class CDataPerturb(ABC):
    @abstractmethod
    def data_perturbation(self, data):
        '''
        - data_perturbation -
        input: data (flat vector of float)

        return a perturbed version of the input data

        '''
        raise NotImplementedError("Subclasses must implement this method")

    def perturb_dataset(self, data_array):
        '''
        - perturb_dataset -
        input: data (2D array of float, each row is a data sample)

        return a perturbed version of the input dataset

        '''
        perturbed_data = []
        for sample in data_array:
            perturbed_sample = self.data_perturbation(sample)
            perturbed_data.append(perturbed_sample)
        return perturbed_data