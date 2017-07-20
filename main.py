#!/usr/bin/env python3

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable

import itertools
import copy
import atexit
import sys
import argparse


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
        atexit.register(plt.show)
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


def multi_target_epoch(common, decoders, optim, X, targets, batch_size=16, shuffle=True, train=True):
    assert len(decoders) == len(targets)
    N = X.size()[0]
    for t in targets:
        assert t.size()[0] == N

    train_X = copy.deepcopy(X)
    train_targets = copy.deepcopy(targets)
    if shuffle:
        p = np.random.permutation(N)
        train_X = train_X.numpy()[p]
        train_X = torch.from_numpy(train_X)
        for i, t in enumerate(train_targets):
            train_targets[i] = torch.from_numpy(t.numpy()[p])


    nb_batches = N // batch_size
    total = nb_batches * batch_size
    cum_losses = []
    nb_correct = []
    for i in range(len(decoders)):
        cum_losses.append(0.0)
        nb_correct.append(0)

    criterion = torch.nn.NLLLoss()

    for i in range(nb_batches):
        batch_X = Variable(train_X[i*batch_size:(i+1)*batch_size])
        batch_ts = []
        for t in train_targets:
            batch_ts.append(Variable(t[i*batch_size:(i+1)*batch_size]))

        repre = common(batch_X) 

        losses = []
        for i, (dec, t) in enumerate(zip(decoders, batch_ts)):
            y = dec(repre)
            loss = criterion(y, t) 
            losses.append(loss)
            _, preds = y.max(dim=1)
            nb_correct[i] += (preds == t).sum().data[0]
            cum_losses[i] += loss.data[0]

        if train:
            complete_loss = sum(losses)
            optim.zero_grad()
            complete_loss.backward()
            optim.step()

    return [cl/nb_batches for cl in cum_losses], [c/total for c in nb_correct]

def instantiate_generators():
    phn_mus = []
    phn_mus.append(np.asarray([1,1]))
    phn_mus.append(np.asarray([3.5,2]))
    phn_mus.append(np.asarray([2,4]))

    phn_covs = []
    phn_covs.append(np.asarray([[0.2,0], [0,0.2]]))
    phn_covs.append(np.asarray([[0.2,0], [0,0.2]]))
    phn_covs.append(np.asarray([[0.2,0], [0,0.2]]))

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

def train(common, decoders, params, X, ts, nb_epochs, report_interval=25):
    optim = torch.optim.SGD(params, lr=1e-3)
    # optim = torch.optim.SGD(itertools.chain(bn_extractor.parameters(), phn_decoder.parameters()), lr=1e-3)
    for i in range(nb_epochs):
        ce, acc = multi_target_epoch(common, decoders, optim, X, ts)
        val_ce, val_acc = multi_target_epoch(common, decoders, optim, X, ts, train=False)

        if i % report_interval == report_interval - 1:
            string = "Train {:>3} CE: {:.3f}, Acc: {:2.2f} | Valid: CE {:.3f}, Acc {:.3f}".format(
                i, ce[0], 100.0*acc[0], val_ce[0], 100.0*val_acc[0]
            )
            print(string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-epochs", type=int, default=200,
                        help="number of training epochs")
    args = parser.parse_args()

    gens = instantiate_generators() 
        
    X, t_phn, t_spk = generate(gens, 100)
    X_val, t_phn_val, t_spk_val = generate(gens, 100)

    plotter = Plotter()
    plotter.plot(X, t_phn, t_spk, name="Raw data")

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
    train(bn_extractor, [phn_decoder],
          itertools.chain(bn_extractor.parameters(), phn_decoder.parameters()),
          X, [t_phn], args.nb_epochs)

    plotter.plot(X, t_phn, t_spk, name="BN features, PHN optimized", transform=bn_extractor)

    spk_decoder = torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,3),
        torch.nn.LogSoftmax()
    )
    spk_backup = copy.deepcopy(spk_decoder)

    print("Training SPK decoder")
    train(bn_extractor, [spk_decoder],
          spk_decoder.parameters(), X, [t_spk], args.nb_epochs)

    print("Training jointly, from same init:")
    optim = torch.optim.SGD(itertools.chain(bn_backup.parameters(), phn_backup.parameters(), spk_backup.parameters()), lr=1e-3)
    for i in range(args.nb_epochs):
        ces, accs= multi_target_epoch(bn_backup, [phn_backup, spk_backup], optim, X, [t_phn, t_spk])
        if i % 25 == 24:
            string = "{:>3} phn CE: {:.3f}, phn Acc: {:2.2f}, spk CE: {:.3f}, spk Acc: {:2.2f}".format(
                i, ces[0], 100.0*accs[0], ces[1], 100.0*accs[1]
            )
            print(string)
    plotter.plot(X, t_phn, t_spk, name="BN features, PHN+SPK optimized", transform=bn_backup)
