import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from base.base_net import BaseNet
from .layers.temporal import TemporalBlock

class TemporalConvNet(BaseNet):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepTCN(BaseNet):

    def __init__(self, num_inputs, num_channels,rep_dim=127, bias = False):
        super().__init__()
        self.num_inputs = num_inputs
        self.rep_dim = rep_dim
        self.tcn = TemporalConvNet(num_inputs=num_inputs,num_channels=num_channels)
        self.fc1 = nn.Linear(num_channels[-1], self.rep_dim, bias=False)


    def forward(self, x):
        x = x.view(-1, self.num_inputs, 1)
        x = self.tcn(x)
        x = self.fc1(x[:,:,-1])
        return x
