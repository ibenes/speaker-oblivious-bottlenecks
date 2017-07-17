#!/usr/bin/env python3

import numpy as np
import torch

import matplotlib.pyplot as plt

class PhnSpkGenerator():
    def __init__(self, mu, cov, phn, spk):
        self._mu = mu
        self._cov = cov
        self._phn = phn
        self._spk = spk

    def generate(self, N):
        X = np.random.multivariate_normal(self._mu, self._cov, size=N)
        phn = torch.ones(N) * self._phn
        spk = torch.ones(N) * self._spk
        return torch.from_numpy(X).float(), phn, spk

if __name__ == '__main__':
    print('hello')

    gen1 = PhnSpkGenerator([4,1], [[2, 0.5],[0.5, 0.5]], phn=0, spk=0)
    X, phn, spk = gen1.generate(100)

    plt.figure()
    plt.scatter(X.numpy()[:,0], X.numpy()[:,1]) 
    plt.show()
