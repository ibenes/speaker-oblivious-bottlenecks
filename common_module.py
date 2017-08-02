import numpy as np
import torch
from torch.autograd import Variable

import atexit
import copy

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


class Plotter():
    def __init__(self, no_plot):
        self._no_plot = no_plot

        if self._no_plot:
            return

        atexit.register(plt.show)
        self._cmap = matplotlib.colors.ListedColormap([
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)
        ])

    def plot(self, X, phn, spk, name="fig", transform = lambda x:x):
        if self._no_plot:
            return self.last_axes_boundaries()

        plt.figure(name)

        for i, m in enumerate(['o', '+', 'x']):
            mask = (spk.numpy() == i)
            spk_set = X.numpy()[mask]
            spk_set = Variable(torch.from_numpy(spk_set).float())
            spk_set = transform(spk_set).data.numpy()
            plt.scatter(spk_set[:,0], spk_set[:,1],
                        c=phn.numpy()[mask], cmap=self._cmap, marker=m) 
        self._show_plot()

        return self.last_axes_boundaries()

    def plot_preds(self, name, X, y, colors):
        if self._no_plot:
            return self.last_axes_boundaries()

        plt.figure(name)
        plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=colors)
        self._show_plot()

        return self.last_axes_boundaries()

    def last_axes_boundaries(self):
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        xmin, xmax = axes.get_xlim()
        
        return (xmin, ymin), (xmax, ymax)

    def _show_plot(self):
        plt.show(block=False)
        plt.pause(0.05)


def plot_preds(plotter, name, bottom_left, upper_right, classifier, nb_steps=100):
    X = np.mgrid[bottom_left[0]:upper_right[0]:(upper_right[0]-bottom_left[0])/nb_steps,
                 bottom_left[1]:upper_right[1]:(upper_right[1]-bottom_left[1])/nb_steps]
    X = X.reshape(2, -1).T
    X = torch.from_numpy(X).float()
    print(X.size())

    y = classifier(Variable(X))
    colors = torch.exp(y).data.numpy()

    plotter.plot_preds(name, X, y, colors)


def create_models(bne_width):
    bn_extractor_init = torch.nn.Sequential(
        torch.nn.Linear(2, bne_width),
        torch.nn.ReLU(),
        torch.nn.Linear(bne_width, bne_width),
        torch.nn.ReLU(),
        torch.nn.Linear(bne_width, 2),
    )
    
    phn_decoder_init = torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10,3),
        torch.nn.LogSoftmax()
    )

    spk_decoder_init = torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10,3),
        torch.nn.LogSoftmax()
    )

    return bn_extractor_init, phn_decoder_init, spk_decoder_init


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
    phn_mus.append(np.asarray([0.7,1.7]))
    phn_mus.append(np.asarray([3.5,2]))
    phn_mus.append(np.asarray([2,4.2]))

    phn_covs = []
    phn_covs.append(np.asarray([[0.5,0], [0,0.2]]))
    phn_covs.append(np.asarray([[0.2,-0.2], [-0.2,0.5]]))
    phn_covs.append(np.asarray([[0.2,-0.15], [-0.15,0.2]]))

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

def grouping_reporter(epoch, lr, losses, accs, val_losses, val_accs):
    string = "{:>3}, lr {:.3e}".format(epoch, lr)
    for l, a in zip(losses, accs):
        string += " ({:.3f} {:.3f})".format(l, a)
    string += " |"
    for l, a in zip(val_losses, val_accs):
        string += " ({:.3f} {:.3f})".format(l, a)
    return string

