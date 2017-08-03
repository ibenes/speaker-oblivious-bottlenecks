#!/usr/bin/env python3

import numpy as np
import torch

import itertools
import argparse

import common_module


def train(common, decoders, params,
          train_data, val_data,
          nb_epochs, report_interval=25,
          reporter=common_module.grouping_reporter):
    lr = 1e-3
    optim = torch.optim.Adam(params, lr=lr)
    best_val_loss = float("inf")
    patience_init = 5

    patience = patience_init

    for i in range(nb_epochs):
        ce, acc = common_module.multi_target_epoch(
            common, decoders, optim,
            train_data[0], train_data[1]
        )
        val_ce, val_acc = common_module.multi_target_epoch(
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


def main(args):
    np.random.seed(args.seed)
    gens = common_module.instantiate_generators()

    X, t_phn, t_spk = common_module.generate(gens, 100)
    X_val, t_phn_val, t_spk_val = common_module.generate(gens, 100)

    plotter = common_module.Plotter(args.no_plot)
    plotter.plot(X, t_phn, t_spk, name="Raw data")
    raw_bl, raw_ur = plotter.plot(
        X_val, t_phn_val, t_spk_val, name="Raw validation data"
    )

    torch.manual_seed(args.seed)
    bne, phn_dec, spk_dec = common_module.create_models(args.bne_width)

    print("\nTraining PHN network")
    train(bne, [phn_dec],
          itertools.chain(bne.parameters(), phn_dec.parameters()),
          (X, [t_phn]), (X_val, [t_phn_val]),
          args.nb_epochs)

    bl, ur = plotter.plot(
        X, t_phn, t_spk,
        name="BN features, PHN optimized", transform=bne
    )
    common_module.plot_preds(
        plotter, "PHN decoding in raw space",
        raw_bl, raw_ur, lambda x: phn_dec(bne(x))
    )
    common_module.plot_preds(
        plotter, "PHN decoding in BN space",
        bl, ur, phn_dec
    )

    print("\nTraining SPK decoder")
    train(bne, [spk_dec],
          spk_dec.parameters(),
          (X, [t_spk]), (X_val, [t_spk_val]),
          args.nb_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument(
        "--bne-width", type=int, default=100,
        help="width of the bottleneck extractor (its hidden layers)"
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="seed for both NumPy data and PyTorch model weights sampling"
    )
    parser.add_argument("--no-plot", action="store_true",
                        help="do no plotting")
    args = parser.parse_args()

    main(args)
