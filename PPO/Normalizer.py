import numpy as np


class Normalizer:
    def __init__(self, dim, absolute_max_value):
        self._n = 0
        if dim == 1:
            self._m = 0.0
            self._s = 0.0
            self._t = 0.0
            self._mean = 0.0
            self._std = 1.0
        else:
            self._m = np.zeros(dim, dtype=float)
            self._s = np.zeros(dim, dtype=float)
            self._t = np.zeros(dim, dtype=float)
            self._mean = np.zeros(dim, dtype=float)
            self._std = np.ones(dim, dtype=float)
        self.absolute_max_value = absolute_max_value
        self.dim = dim

    def add(self, x):
        if self.dim > 1 and not isinstance(x, np.floating):
            x = np.array(x, dtype=float)

        if self._n == 0:
            self._n = 1
            self._t = x.copy() if self.dim > 1 else x
            self._mean = x.copy() if self.dim > 1 else x
        else:
            self._n += 1

            self._t += x
            previous_mean = self._mean.copy() if self.dim > 1 else self._mean
            self._mean = self._t / self._n
            self._m = self._m + (x-previous_mean) * (x-self._mean)
            self._std = np.sqrt(self._m / (self._n))

    def normalize(self, x):
        if self.dim > 1 and not isinstance(x, np.floating):
            x = np.array(x, dtype=float)

        return np.clip(
            (x-self._mean) / (self._std+1e-6),
            -self.absolute_max_value,
            self.absolute_max_value)

    def add_and_normalize(self, x):
        if self.dim > 1 and not isinstance(x, np.floating):
            x = np.array(x, dtype=float)

        self.add(x)
        return self.normalize(x)
