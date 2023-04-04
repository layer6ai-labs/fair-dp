import math

import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class MLP(nn.Module):

    def __init__(self, n_units_list, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev_layer_size = n_units_list[0]
        for n_units in n_units_list[1:-1]:
            layers.append(nn.Linear(in_features=prev_layer_size, out_features=n_units))
            prev_layer_size = n_units
            layers.append(activation())
        layers.append(nn.Linear(in_features=prev_layer_size, out_features=n_units_list[-1]))
        self.net = nn.Sequential(*layers)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(
            self,
            input_channels,
            hidden_channels_list,
            output_dim,
            kernel_size,
            stride,
            image_height,
            activation=nn.ReLU,
    ):
        super().__init__()
        if type(stride) not in [list, tuple]:
            stride = [stride for _ in hidden_channels_list]

        if type(kernel_size) not in [list, tuple]:
            kernel_size = [kernel_size for _ in hidden_channels_list]

        cnn_layers = []
        prev_channels = input_channels
        for hidden_channels, k, s in zip(hidden_channels_list, kernel_size, stride):
            cnn_layers.append(nn.Conv2d(prev_channels, hidden_channels, k, s))
            cnn_layers.append(activation())
            prev_channels = hidden_channels

            # NOTE: Assumes square image
            image_height = self._get_new_image_height(image_height, k, s)
        self.cnn_layers = nn.ModuleList(cnn_layers)

        self.fc_layer = nn.Linear(prev_channels * image_height ** 2, output_dim)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, x):
        for layer in self.cnn_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)

        return self.fc_layer(x)

    def _get_new_image_height(self, height, kernel, stride):
        # cf. https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv2d.html
        # Assume dilation = 1, padding = 0
        return math.floor((height - kernel) / stride + 1)
