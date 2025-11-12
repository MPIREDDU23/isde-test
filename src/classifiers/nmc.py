import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self.centroids = None
        self.class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids
    @centroids.setter
    def centroids(self, value):
        self._centroids = value

    @property
    def class_labels(self):
        return self._class_labels
    @class_labels.setter
    def class_labels(self, value):
        self._class_labels = value

    def fit(self, xtr, ytr):
        '''
        Compute the averege of each centroid
        '''
        # Identify unique class labels
        self._class_labels = np.unique(ytr)

        # Number of classes is the number of unique labels
        n_classes = self.class_labels.size

        # Number of features is the number of columns in xtr
        n_features = xtr.shape[1]

        # Initialize centroids array
        self.centroids = np.zeros((n_classes, n_features))

        # Compute centroids for each class
        for idx, label in enumerate(self._class_labels):
            # Get all samples belonging to the current class
            class_samples = xtr[ytr == label]

            # Compute the mean of the samples for the current class
            self.centroids[idx] = np.mean(class_samples, axis=0)

    def predict(self, Xts):
        if self.centroids is None:
            raise ValueError("The classifier is not trained. Call fit first!")

        dist_euclidean = euclidean_distances(Xts, self.centroids)
        idx_min = np.argmin(dist_euclidean, axis=1)
        yc = self.class_labels[idx_min]
        return yc