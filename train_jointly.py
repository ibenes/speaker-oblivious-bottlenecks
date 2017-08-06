#!/usr/bin/env python3

import numpy as np
import torch

import itertools
import argparse

import training
import data
import plotting
import model


def main(args):
    np.random.seed(args.seed)
    gens = data.instantiate_generators()

    X, t_phn, t_spk = data.generate(gens, 100)
    X_val, t_phn_val, t_spk_val = data.generate(gens, 100)

    plotter = plotting.Plotter(args.no_plot)
    plotter.plot(X, t_phn, t_spk, name="Raw data")
    raw_bl, raw_ur = plotter.plot(
        X_val, t_phn_val,
        t_spk_val, name="Raw validation data"
    )

    torch.manual_seed(args.seed)
    bne, phn_dec, spk_dec = model.create_models(args.bne_width)

    training.train(
        bne, [phn_dec, spk_dec],
        itertools.chain(
            bne.parameters(), phn_dec.parameters(), spk_dec.parameters()
        ),
        (X, [t_phn, t_spk]), (X_val, [t_phn_val, t_spk_val]),
        args.nb_epochs
    )

    bl, ur = plotter.plot(
        X, t_phn, t_spk,
        name="BN features, PHN+SPK optimized", transform=bne
    )
    plotting.plot_preds(
        plotter, "PHN decoding in jointly trained BN space",
        bl, ur, phn_dec
    )


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
