import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.conv1 = GCNConv(self.input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return x


class NodeNormalizer:
    def __init__(self, graphs):
        stats_1 = torch.zeros_like(graphs[0].node_features[0])
        stats_2 = torch.zeros_like(graphs[0].node_features[0])
        nb_nodes = 0

        for g in graphs:
            stats_1 += g.node_features.sum(axis=0)
            stats_2 += g.node_features.pow(2).sum(axis=0)
            nb_nodes += g.node_features.shape[0]

        self.mu = stats_1 / nb_nodes
        self.std = (stats_2 / nb_nodes - self.mu ** 2) ** 0.5

    def normalize_graphs(self, graphs):
        for g in graphs:
            g.node_features = (g.node_features - self.mu) / self.std
