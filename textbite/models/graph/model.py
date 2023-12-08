from typing import Callable

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn


class GraphModel(torch.nn.Module):
    def __init__(
        self,
        device,
        input_size: int,
        n_layers: int = 1,
        hidden_size: int = 128,
        activation: Callable = nn.ReLU(),
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.input_size = input_size

        self.conv1 = GCNConv(self.input_size, 97)
        self.conv2 = GCNConv(97, 97)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x
