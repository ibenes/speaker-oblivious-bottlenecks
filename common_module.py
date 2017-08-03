import numpy as np
import torch
from torch.autograd import Variable

import atexit
import copy

import matplotlib
import matplotlib.pyplot as plt


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

    def plot(self, X, phn, spk, name="fig", transform=lambda x: x):
        if self._no_plot:
            return self.last_axes_boundaries()

        plt.figure(name)

        for i, m in enumerate(['o', '+', 'x']):
            mask = (spk.numpy() == i)
            spk_set = X.numpy()[mask]
            spk_set = Variable(torch.from_numpy(spk_set).float())
            spk_set = transform(spk_set).data.numpy()
            plt.scatter(spk_set[:, 0], spk_set[:, 1],
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


def plot_preds(plotter, name, b_l, u_r,
               classifier, nb_steps=100):
    X = np.mgrid[b_l[0]:u_r[0]:(u_r[0]-b_l[0])/nb_steps,
                 b_l[1]:u_r[1]:(u_r[1]-b_l[1])/nb_steps]
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
        torch.nn.Linear(2, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 3),
        torch.nn.LogSoftmax()
    )

    spk_decoder_init = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 3),
        torch.nn.LogSoftmax()
    )

    return bn_extractor_init, phn_decoder_init, spk_decoder_init


def multi_target_epoch(common, decoders, optim, X, targets,
                       batch_size=16, shuffle=True, train=True):
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


def grouping_reporter(epoch, lr, losses, accs, val_losses, val_accs):
    string = "{:>3}, lr {:.3e}".format(epoch, lr)
    for l, a in zip(losses, accs):
        string += " ({:.3f} {:.3f})".format(l, a)
    string += " |"
    for l, a in zip(val_losses, val_accs):
        string += " ({:.3f} {:.3f})".format(l, a)
    return string


def train(common, decoders, params,
          train_data, val_data,
          nb_epochs, report_interval=25,
          reporter=grouping_reporter):
    lr = 1e-3
    optim = torch.optim.Adam(params, lr=lr)
    best_val_loss = float("inf")
    patience_init = 5

    patience = patience_init

    for i in range(nb_epochs):
        ce, acc = multi_target_epoch(
            common, decoders, optim,
            train_data[0], train_data[1]
        )
        val_ce, val_acc = multi_target_epoch(
            common, decoders, optim,
            val_data[0], val_data[1], train=False
        )

        val_loss = sum(val_ce)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = patience_init
        else:
            if patience == 0:
                for param_group in optim.param_groups:
                    lr *= 0.5
                    param_group['lr'] = lr
                string = reporter(i, lr, ce, acc, val_ce, val_acc)
                print(string)
            else:
                patience -= 1

        if lr < 1e-7:
            print("stopping training, because of LR being effectively zero")
            string = reporter(i, lr, ce, acc, val_ce, val_acc)
            print(string)
            break

        if i % report_interval == report_interval - 1:
            string = reporter(i, lr, ce, acc, val_ce, val_acc)
            print(string)
