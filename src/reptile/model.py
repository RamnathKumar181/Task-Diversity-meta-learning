import torch.nn as nn
import torch
from torch.autograd import Variable

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d,
                               MetaSequential, MetaLinear)


def conv_block(in_channels, out_channels, max_pool=False, **kwargs):
    if max_pool:
        return MetaSequential(OrderedDict([
            ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
            ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
                                    track_running_stats=False)),
            ('pool', nn.MaxPool2d(kernel_size=2)),
            ('relu', nn.ReLU())
        ]))

    else:
        return MetaSequential(OrderedDict([
            ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
            ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
                                    track_running_stats=False)),
            ('relu', nn.ReLU())
        ]))


class MetaConvModel(nn.Module):
    """4-layer Convolutional Neural Network architecture from [1].
    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.
    out_features : int
        Number of classes (output of the model).
    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.
    feature_size : int (default: 64)
        Number of features returned by the convolutional head.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML)
           (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, in_channels, out_features,
                 hidden_size=64, feature_size=64, use_max_pool=False):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        if use_max_pool:
            self.features = MetaSequential(OrderedDict([
                ('layer1', conv_block(in_channels, hidden_size, max_pool=use_max_pool, kernel_size=3,
                                      stride=2, padding=1, bias=True)),
                ('layer2', conv_block(hidden_size, hidden_size, max_pool=use_max_pool, kernel_size=3,
                                      stride=2, padding=1, bias=True)),
                ('layer3', conv_block(hidden_size, hidden_size, max_pool=use_max_pool, kernel_size=3,
                                      stride=2, padding=1, bias=True)),
                ('layer4', conv_block(hidden_size, hidden_size, max_pool=use_max_pool, kernel_size=3,
                                      stride=2, padding=1, bias=True))
            ]))
        else:
            self.features = nn.Sequential(
                # 28 x 28 - 1
                nn.Conv2d(1, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                # 14 x 14 - 64
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                # 7 x 7 - 64
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                # 4 x 4 - 64
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                # 2 x 2 - 64
            )
        self.classifier = nn.Sequential(
            # 2 x 2 x 64 = 256
            nn.Linear(256, self.out_features),
            nn.LogSoftmax(1)
        )

    def forward(self, inputs):
        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        logits = self.classifier(features)
        return logits


class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].
    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of classes (output of the model).
    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML)
           (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
                                                     MetaSequential(OrderedDict([
                                                         ('linear', MetaLinear(hidden_size,
                                                          layer_sizes[i + 1], bias=True)),
                                                         ('relu', nn.ReLU())
                                                     ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params,
                                                                 'features'))
        logits = self.classifier(features,
                                 params=self.get_subdict(params, 'classifier'))
        return logits


def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size)


def ModelConvMiniImagenet(out_features, hidden_size=32):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size*5*5, use_max_pool=True)


if __name__ == '__main__':
    model = ModelConvOmniglot()
