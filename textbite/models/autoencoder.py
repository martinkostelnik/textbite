from __future__ import annotations
import torch
from torch import nn

from textbite.models.utils import ModelType
    

class AutoEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            encoding_size: int,
            device,
        ):
        super().__init__()

        self.model_type = ModelType.AE
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.device = device

        self.encoder = nn.Sequential(
            torch.nn.Linear(input_size, 60), 
            torch.nn.ReLU(), 
            torch.nn.Linear(60, 40), 
            torch.nn.ReLU(),
            torch.nn.Linear(40, encoding_size),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_size, 40), 
            torch.nn.ReLU(), 
            torch.nn.Linear(40, 60), 
            torch.nn.ReLU(), 
            torch.nn.Linear(60, input_size),
        )

    def forward(self, x):
        encoding = self.encoder(x)
        reconstructed = self.decoder(encoding)

        return encoding, reconstructed

    @property
    def dict_for_saving(self):
        dict_for_saving = {
            "state_dict": self.state_dict(),
            "input_size": self.input_size,
            "encoding_size": self.encoding_size,
            "model_type": self.model_type,
        }

        return dict_for_saving

    def from_pretrained(checkpoint: dict, device) -> AutoEncoder:
        model = AutoEncoder(
            input_size=checkpoint["input_size"],
            encoding_size=checkpoint["encoding_size"],
            device=device,
        )

        model.load_state_dict(checkpoint["state_dict"])
        return model
