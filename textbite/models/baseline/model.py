from typing import Callable

from torch import nn


class BaselineModel(nn.Module):
    def __init__(
        self,
        device,
        n_layers: int = 1,
        hidden_size: int = 128,
        activation: Callable = nn.ReLU(),
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout_prob = dropout_prob

        self.input_size = 97
        self.output_size = 3

        self.layers = nn.ModuleList()

        if n_layers == 1:
            self.layers.append(nn.Linear(self.input_size, self.output_size))
        else:
            self.layers.append(nn.Linear(self.input_size, self.hidden_size))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(self.dropout_prob))

            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                self.layers.append(self.activation)
                self.layers.append(nn.Dropout(self.dropout_prob))

            self.layers.append(nn.Linear(self.hidden_size, self.output_size))

    def forward(self, features):
        for layer in self.layers:
            features = layer(features)

        return features
