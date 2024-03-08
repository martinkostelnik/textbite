import torch
from torch import nn
from torch_geometric.nn import ResGatedGraphConv


class Block(torch.nn.Module):
    def __init__(
        self,
        width,
        dropout_prob,
    ):
        super().__init__()
        self.gcn = ResGatedGraphConv(width, width, edge_dim=1)
        self.norm = nn.LayerNorm(width)
        self.nonlin = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, edge_index, edge_attr):
        x = self.gcn(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        x = self.nonlin(x)
        # x = self.dropout(x)

        return x


class JoinerGraphModel(torch.nn.Module):
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
        self.dropout = nn.Dropout(dropout_prob)

        self.in_proj = torch.nn.Linear(self.input_size, self.hidden_size)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(Block(self.hidden_size, dropout_prob))
        self.out_proj = torch.nn.Linear(self.hidden_size, self.output_size, bias=False)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = edge_attr.reshape(shape=(len(edge_attr), 1))
        x = self.in_proj(x)
        x = self.dropout(x)
        for block in self.blocks:
            # x = x + block(x, edge_index, edge_attr)
            x = block(x, edge_index, edge_attr)
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
