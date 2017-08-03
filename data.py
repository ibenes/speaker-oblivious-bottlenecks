import numpy as np
import torch


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


def instantiate_generators():
    phn_mus = []
    phn_mus.append(np.asarray([0.7, 1.7]))
    phn_mus.append(np.asarray([3.5, 2]))
    phn_mus.append(np.asarray([2, 4.2]))

    phn_covs = []
    phn_covs.append(np.asarray([[0.5,  0],    [0,     0.2]]))
    phn_covs.append(np.asarray([[0.2, -0.2],  [-0.2,  0.5]]))
    phn_covs.append(np.asarray([[0.2, -0.15], [-0.15, 0.2]]))

    spk_mus = []
    spk_mus.append(np.asarray([-0.7, 0.7]))
    spk_mus.append(np.asarray([1, 0]))
    spk_mus.append(np.asarray([-0.7, -0.7]))

    gens = []
    for phn, (phn_mu, phn_cov) in enumerate(zip(phn_mus, phn_covs)):
        for spk, spk_mu in enumerate(spk_mus):
            gens.append(PhnSpkGenerator(phn_mu+spk_mu, phn_cov, phn, spk))

    return gens


def generate(gens, N_per_cluster):
    X = torch.zeros((0, 2))
    t_phn = torch.zeros((0,))
    t_spk = torch.zeros((0,))

    for g in gens:
        X_g, phn_g, spk_g = g.generate(100)
        X = torch.cat([X, X_g], 0)
        t_phn = torch.cat([t_phn, phn_g], 0)
        t_spk = torch.cat([t_spk, spk_g], 0)
    t_phn = t_phn.long()
    t_spk = t_spk.long()

    return X, t_phn, t_spk
