#!/usr/bin/env python3

import numpy as np
import torch

import matplotlib
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
    phn_mus = []
    phn_mus.append(np.asarray([1,1]))
    phn_mus.append(np.asarray([3,-2]))
    phn_mus.append(np.asarray([6,4]))

    phn_covs = []
    phn_covs.append(np.asarray([[1,0], [0,1]]))
    phn_covs.append(np.asarray([[1,0], [0,1]]))
    phn_covs.append(np.asarray([[1,0], [0,1]]))

    spk_mus = []
    spk_mus.append(np.asarray([0, 3]))
    spk_mus.append(np.asarray([0, 6]))
    spk_mus.append(np.asarray([0, 9]))

    gens = []
    for phn, (phn_mu, phn_cov) in enumerate(zip(phn_mus, phn_covs)):
        for spk, spk_mu in enumerate(spk_mus):
            gens.append(PhnSpkGenerator(phn_mu+spk_mu, phn_cov, phn, spk))
        
    X = torch.zeros((0, 2))
    t_phn = torch.zeros((0,))
    t_spk = torch.zeros((0,))

    for g in gens:
        X_g, phn_g, spk_g = g.generate(100)
        X = torch.cat([X, X_g], 0)
        t_phn = torch.cat([t_phn, phn_g], 0)
        t_spk = torch.cat([t_spk, spk_g], 0)

    my_colors = matplotlib.colors.ListedColormap([
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1)
    ])

    plt.figure()
    plt.scatter(X.numpy()[:,0], X.numpy()[:,1], c=t_phn.numpy(), cmap=my_colors) 
    plt.show()
