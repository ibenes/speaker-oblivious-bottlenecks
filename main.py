#!/usr/bin/env python3

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable

from itertools import chain
import atexit


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


class Plotter():
    def __init__(self):
        self._cmap = matplotlib.colors.ListedColormap([
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)
        ])

    def plot(self, X, phn, spk, name="fig", transform = lambda x:x):
        plt.figure(name)

        for i, m in enumerate(['o', '+', 'x']):
            mask = (spk.numpy() == i)
            spk_set = X.numpy()[mask]
            spk_set = Variable(torch.from_numpy(spk_set).float())
            spk_set = transform(spk_set).data.numpy()
            plt.scatter(spk_set[:,0], spk_set[:,1],
                        c=t_phn.numpy()[mask], cmap=self._cmap, marker=m) 
        plt.show(block=False)


def epoch(fwd, params, X, target, batch_size=16, shuffle=True):
    N = X.size()[0]
    assert target.size()[0] == N

    train_X = X
    train_t = target
    if shuffle:
        p = np.random.permutation(N)
        train_X = X.numpy()[p]
        train_t = target.numpy()[p]

    train_X = torch.from_numpy(train_X)
    train_t = torch.from_numpy(train_t)

    nb_batches = N // batch_size
    total = nb_batches * batch_size
    total_loss = 0.0
    correct = 0

    criterion = torch.nn.NLLLoss()
    optim = torch.optim.SGD(params, lr=1e-3)

    for i in range(nb_batches):
        batch_X = Variable(train_X[i*batch_size:(i+1)*batch_size])
        batch_t = Variable(train_t[i*batch_size:(i+1)*batch_size])

        y = fwd(batch_X)
        loss = criterion(y, batch_t) 

        total_loss += loss.data[0] 

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_loss/nb_batches, correct/total
        

if __name__ == '__main__':
    atexit.register(plt.show)
     
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
    t_phn = t_phn.long()
    t_spk = t_spk.long()

    plotter = Plotter()
    plotter.plot(X, t_phn, t_spk, name="Raw data")

    bn_extractor = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2),
    )

    plotter.plot(X, t_phn, t_spk, name="BN features, random init", transform=bn_extractor)

    phn_decoder = torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,3),
        torch.nn.LogSoftmax()
    )

    for i in range(100):
        ce, acc = epoch(lambda x: phn_decoder(bn_extractor(x)),
                        chain(bn_extractor.parameters(), phn_decoder.parameters()),
                        X, t_phn)
        print(i, "CE:", ce, "Acc:", acc)

    plotter.plot(X, t_phn, t_spk, name="BN features, PHN optimized", transform=bn_extractor)
