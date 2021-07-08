import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Protonet(nn.Module):
    """Prototypical Network architecture from [1].
    Parameters
    ----------
    x_dim : int
        Number of channels for the input images.
    z_dim : int
        Dimensions of the output embedding (output of the model).
    hid_dim : int (default: 64)
        Number of channels in the intermediate representations.
    References
    ----------
    .. [1] Snell, Jake, Kevin Swersky, and Richard Zemel.
           "Prototypical networks for few-shot learning." Proceedings of the
           31st International Conference on Neural Information Processing
           Systems. 2017. (https://arxiv.org/abs/1703.05175)
    """

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def Protonet_Omniglot(out_features=64, hidden_size=64):
    return Protonet(1, hid_dim=hidden_size, z_dim=out_features)


def Protonet_MiniImagenet(out_features=1600, hidden_size=64):
    return Protonet(3, hid_dim=hidden_size, z_dim=out_features)
