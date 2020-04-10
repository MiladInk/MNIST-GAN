from typing import List

import matplotlib.pyplot as plt
from numpy import random


class RandomVariable:

    @staticmethod
    def plot_samples(samples, bins) -> None:
        plt.hist(samples, bins=bins, density=True)
        plt.show()

    def __int__(self):
        pass

    def sample(self, size: int = 1) -> List[float]:
        raise NotImplementedError

    def plot(self, sample_size: int = 1000, bins: int = 20) -> None:
        samples = self.sample(sample_size)
        return RandomVariable.plot_samples(samples=samples, bins=bins)


class NormalVariable(RandomVariable):

    def __init__(self, mean: float = 1, std: float = 1) -> None:
        super(NormalVariable, self).__init__()
        self.mean = mean
        self.std = std

    def sample(self, size: int = 1) -> List[float]:
        return random.normal(self.mean, self.std, size=size)


n1 = NormalVariable(0, 1)
print(n1.sample()[0])
print(n1.sample()[0])
n1.plot(sample_size=10000, bins=40)
samples = NormalVariable(3, 4).sample(10000)
samples = [(s - 3) / 4 for s in samples]
RandomVariable.plot_samples(samples, bins=40)
