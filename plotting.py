import numpy as np
import torch
from torch.autograd import Variable

import atexit

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
