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

    phn_mus = []
    phn_mus.append(np.asarray([1,1]))
    phn_mus.append(np.asarray([3,-2]))
    phn_mus.append(np.asarray([6,4]))

    phn_covs = []
    phn_covs.append(np.asarray([[1,0], [0,1]]))
    phn_covs.append(np.asarray([[1,0], [0,1]]))
    phn_covs.append(np.asarray([[1,0], [0,1]]))

    gen1 = PhnSpkGenerator([4,1], [[2, 0.5],[0.5, 0.5]], phn=0, spk=0)
    X, phn, spk = gen1.generate(100)

    gens = []
    for phn_mu, phn_cov in zip(phn_mus, phn_covs):
        gens.append(PhnSpkGenerator(phn_mu, phn_cov, 0, 0))
        
    X = torch.zeros((0, 2))
    t_phn = torch.zeros((0,))
    t_spk = torch.zeros((0,))

    for g in gens:
        X_g, phn_g, spk_g = g.generate(100)
        X = torch.cat([X, X_g], 0)
        t_phn = torch.cat([t_phn, phn_g], 0)
        t_spk = torch.cat([t_spk, spk_g], 0)

    plt.figure()
    plt.scatter(X.numpy()[:,0], X.numpy()[:,1]) 
    plt.show()
