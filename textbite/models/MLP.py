from __future__ import annotations


"""MLP definition

Date -- 15.05.2024
Author -- Martin Kostelnik

"""

from typing import Callable

from torch import nn

from textbite.models.utils import ModelType


class MLP(nn.Module):
    def __init__(
        self,
        device,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int = 1,
        activation: Callable = nn.GELU(),
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.model_type = ModelType.MLP

        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout_prob = dropout_prob

        self.input_size = input_size
        self.output_size = output_size

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
    
    @property
    def dict_for_saving(self):
        dict_for_saving = {
            "state_dict": self.state_dict(),
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_layers": self.n_layers,
            "dropout_prob": self.dropout_prob,
            "model_type": self.model_type,
        }

        return dict_for_saving

    def from_pretrained(checkpoint: dict, device) -> MLP:
        mlp = MLP(
            device = device,
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            output_size=checkpoint["output_size"],
            n_layers=checkpoint["n_layers"]
        )

        mlp.load_state_dict(checkpoint["state_dict"])
        return mlp
