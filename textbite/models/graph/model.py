from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
        x = self.gcn(x, edge_index)
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
        self.dropout_prob = dropout_prob

        self.in_proj = torch.nn.Linear(self.input_size, self.hidden_size)
        self.out_proj = torch.nn.Linear(self.hidden_size, self.output_size)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(Block(self.hidden_size, dropout_prob))

        # self.layers = nn.ModuleList()
        # if n_layers == 1:
        #     self.layers.append(GCNConv(self.input_size, self.output_size))
        # else:
        #     self.layers.append(GCNConv(self.input_size, hidden_size))
        #     self.layers.append(nn.ReLU())
        #     self.layers.append(nn.Dropout(self.dropout_prob))
        #
        #     for _ in range(n_layers - 2):
        #         self.layers.append(GCNConv(hidden_size, hidden_size))
        #         self.layers.append(nn.ReLU())
        #         self.layers.append(nn.Dropout(self.dropout_prob))
        #
        #     self.layers.append(GCNConv(hidden_size, output_size))

    def forward(self, x, edge_index):
        x = self.in_proj(x)
        for block in self.blocks:
            x = x + block(x, edge_index)
        x = self.out_proj(x)

        return x


class NodeNormalizer:
    def __init__(self, graphs: List[Graph]):
        stats_1 = torch.zeros_like(graphs[0].node_features[0])
        stats_2 = torch.zeros_like(graphs[0].node_features[0])
        nb_nodes = 0

        for g in graphs:
            stats_1 += g.node_features.sum(axis=0)
            stats_2 += g.node_features.pow(2).sum(axis=0)
            nb_nodes += g.node_features.shape[0]

        self.mu = stats_1 / nb_nodes
        self.std = (stats_2 / nb_nodes - self.mu ** 2) ** 0.5

    def normalize_graphs(self, graphs: List[Graph]) -> None:
        for g in graphs:
            g.node_features = (g.node_features - self.mu) / self.std
