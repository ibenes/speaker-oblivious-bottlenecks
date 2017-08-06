#!/usr/bin/env python3

import numpy as np
import torch
from torch.autograd import Variable

import itertools
import copy
import sys
import argparse

import training
import data
import plotting
import model


class GradReverter(torch.autograd.Function):
    def forward(self, x):
        return x

    def backward(self, g):
        return -g


def adversary_train(bne, main, adversaly_aux, train_data, val_data, nb_epochs, report_interval=25,
        reporter=training.grouping_reporter):
    lr = 1e-3
    main_optim = torch.optim.Adam(itertools.chain(bne.parameters(), main.parameters()), lr=lr)
    adversary_optim = torch.optim.Adam(adversaly_aux.parameters(), lr=lr)
    best_val_loss = float("inf")

    grad_reverter = GradReverter()
    adversary = lambda x: adversaly_aux(grad_reverter(x))

    patience_init = 5
    patience = patience_init
    adversary_epochs = 3

    for i in range(nb_epochs):
        if lr < 1e-7:
            print("stopping training, because of LR being effectively zero")
            string = reporter(i, lr, ce, acc, val_ce, val_acc)
            print(string)
            break

        ce, acc = training.multi_target_epoch(bne, [main, adversary], main_optim, train_data[0], train_data[1])
        for j in range(adversary_epochs):
            ce, acc = training.multi_target_epoch(bne, [adversary], adversary_optim, train_data[0], train_data[1][1:])

        val_ce, val_acc = training.multi_target_epoch(bne, [main, adversary], main_optim, val_data[0], val_data[1], train=False)

        val_loss = val_ce[0] # we only measure the main (PHN) crossentropy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = patience_init
        else:
            if patience == 0:
                for param_group in itertools.chain(main_optim.param_groups, adversary_optim.param_groups):
                    lr *= 0.5
                    param_group['lr'] = lr
                string = reporter(i, lr, ce, acc, val_ce, val_acc)
                print(string)
            else:
                patience -= 1


        if i % report_interval == report_interval - 1:
            string = reporter(i, lr, ce, acc, val_ce, val_acc)
            print(string)


def main(args):
    np.random.seed(args.seed)
    gens = data.instantiate_generators()

    X, t_phn, t_spk = data.generate(gens, 100)
    X_val, t_phn_val, t_spk_val = data.generate(gens, 100)

    plotter = plotting.Plotter(args.no_plot)
    plotter.plot(X, t_phn, t_spk, name="Raw data")
    raw_bl, raw_ur = plotter.plot(X_val, t_phn_val, t_spk_val, name="Raw validation data")

    torch.manual_seed(args.seed)
    bn_extractor_init, phn_decoder_init, spk_decoder_init = model.create_models(args.bne_width)

    bn_extractor = copy.deepcopy(bn_extractor_init)
    spk_decoder = copy.deepcopy(spk_decoder_init)
    phn_decoder = copy.deepcopy(phn_decoder_init)

    print("\nTraining in disconcert, from same init:")
    adversary_train(bn_extractor, phn_decoder, spk_decoder,
          (X, [t_phn, t_spk]), (X_val, [t_phn_val, t_spk_val]),
          args.nb_epochs)

    bl, ur = plotter.plot(X, t_phn, t_spk, name="BN features, PHN-SPK optimized", transform=bn_extractor)
    plotting.plot_preds(plotter, "PHN decoding in disconcertly trained BN space", bl, ur, phn_decoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--bne-width", type=int, default=100,
                        help="width of the bottleneck extractor (its hidden layers)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="seed for both NumPy data and PyTorch model weights sampling")
    parser.add_argument("--no-plot", action="store_true",
                        help="do no plotting")
    args = parser.parse_args()

    main(args)
