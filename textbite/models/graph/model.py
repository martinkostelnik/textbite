from typing import List

import torch
from torch import nn
from torch_geometric.nn import GCNConv, ResGatedGraphConv

from textbite.models.graph.create_graphs import Graph


class Block(torch.nn.Module):
    def __init__(
        self,
        width,
        dropout_prob,
    ):
        super().__init__()
        self.gcn = GCNConv(width, width)
        self.norm = nn.LayerNorm(width)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index=edge_index)
        x = self.norm(x)
        x = self.nonlin(x)
        x = self.dropout(x)

        return x


class GraphModel(torch.nn.Module):
    def __init__(
        self,
        device,
        input_size: int,
        output_size: int,
        n_layers: int = 3,
        hidden_size: int = 128,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.in_proj = torch.nn.Linear(self.input_size, self.hidden_size)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(Block(self.hidden_size, dropout_prob))
        self.out_proj = torch.nn.Linear(self.hidden_size, self.output_size, bias=False)

    def forward(self, x, edge_index):
        x = self.in_proj(x)
        for block in self.blocks:
            x = x + block(x, edge_index)
        x = self.out_proj(x)

        return x

    @property
    def dict_for_saving(self):
        dict_for_saving = {
            "state_dict": self.state_dict(),
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_layers": self.n_layers,
            "dropout_prob": self.dropout_prob,
        }

        return dict_for_saving


def load_gcn(fn, device):
    model_checkpoint = torch.load(fn, map_location=device)
    model = GraphModel(
        device=device,
        hidden_size=model_checkpoint["hidden_size"],
        input_size=model_checkpoint["input_size"],
        output_size=model_checkpoint["output_size"],
        n_layers=model_checkpoint["n_layers"],
        dropout_prob=model_checkpoint["dropout_prob"],
    )
    model.load_state_dict(model_checkpoint["state_dict"])
    model.to(device)

    return model
