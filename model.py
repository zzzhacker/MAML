import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict
import math

def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self,in_channels, input_dim, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        i = 4
        idim = input_dim
        for x in range(i):
            idim = math.floor(idim/2)

        self.linear_input = hidden_size*idim*idim

        self.classifier = MetaLinear(self.linear_input, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits
