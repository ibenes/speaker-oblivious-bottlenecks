#!/usr/bin/env python3

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable

from itertools import chain
import copy
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
        _, preds = y.max(dim=1)
        correct += (preds == batch_t).sum().data[0]

        total_loss += loss.data[0] 

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_loss/nb_batches, correct/total
        

def dual_target_epoch(common, dec1, dec2, params, X, t1, t2, batch_size=16, shuffle=True):
    N = X.size()[0]
    assert t1.size()[0] == N
    assert t2.size()[0] == N

    train_X = X
    train_t1 = t1
    train_t2 = t2
    if shuffle:
        p = np.random.permutation(N)
        train_X = X.numpy()[p]
        train_t1 = t1.numpy()[p]
        train_t2 = t2.numpy()[p]

    train_X = torch.from_numpy(train_X)
    train_t1 = torch.from_numpy(train_t1)
    train_t2 = torch.from_numpy(train_t2)

    nb_batches = N // batch_size
    total = nb_batches * batch_size
    total_loss1 = 0.0
    total_loss2 = 0.0
    correct1 = 0
    correct2 = 0

    criterion = torch.nn.NLLLoss()
    optim = torch.optim.SGD(params, lr=1e-3)

    for i in range(nb_batches):
        batch_X = Variable(train_X[i*batch_size:(i+1)*batch_size])
        batch_t1 = Variable(train_t1[i*batch_size:(i+1)*batch_size])
        batch_t2 = Variable(train_t2[i*batch_size:(i+1)*batch_size])

        repre = common(batch_X) 

        y1 = dec1(repre)
        loss1 = criterion(y1, batch_t1) 
        _, preds1 = y1.max(dim=1)
        correct1 += (preds1 == batch_t1).sum().data[0]
        total_loss1 += loss1.data[0] 

        y2 = dec2(repre)
        loss2 = criterion(y2, batch_t2) 
        _, preds2 = y2.max(dim=1)
        correct2 += (preds2 == batch_t2).sum().data[0]
        total_loss2 += loss2.data[0] 

        complete_loss = loss1 + loss2

        optim.zero_grad()
        complete_loss.backward()
        optim.step()

    return total_loss1/nb_batches, correct1/total, total_loss2/nb_batches, correct2/total


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
    # plotter.plot(X, t_phn, t_spk, name="Raw data")

    bn_extractor = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2),
    )

    bn_backup = copy.deepcopy(bn_extractor)

    plotter.plot(X, t_phn, t_spk, name="BN features, random init", transform=bn_extractor)

    phn_decoder = torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,3),
        torch.nn.LogSoftmax()
    )
    phn_backup = copy.deepcopy(phn_decoder)

    print("Training PHN network")
    for i in range(200):
        ce, acc = epoch(lambda x: phn_decoder(bn_extractor(x)),
                        chain(bn_extractor.parameters(), phn_decoder.parameters()),
                        X, t_phn)
        if i % 25 == 24:
            print(i, "CE:", ce, "Acc:", acc)

    plotter.plot(X, t_phn, t_spk, name="BN features, PHN optimized", transform=bn_extractor)

    spk_decoder = torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,3),
        torch.nn.LogSoftmax()
    )
    spk_backup = copy.deepcopy(spk_decoder)


    print("Training SPK decoder")
    for i in range(200):
        ce, acc = epoch(lambda x: spk_decoder(bn_extractor(x)),
                        spk_decoder.parameters(),
                        X, t_spk)
        if i % 25 == 24:
            print(i, "CE:", ce, "Acc:", acc)

    print("Training jointly, from same init:")
    for i in range(200):
        phn_ce, phn_acc, spk_ce, spk_acc = dual_target_epoch(
            bn_backup, phn_backup, spk_backup,
            chain(bn_backup.parameters(), phn_backup.parameters(), spk_backup.parameters()),
            X, t_phn, t_spk
        )
        if i % 25 == 24:
            print(i, "phn CE:", phn_ce, "phn Acc:", phn_acc, "spk CE:", spk_ce, "spk Acc:", spk_acc)
    plotter.plot(X, t_phn, t_spk, name="BN features, PHN+SPK optimized", transform=bn_backup)
