import torch


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
