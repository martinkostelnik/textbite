from typing import List
import pickle

import torch

from textbite.models.baseline.utils import Sample


class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.data: List[Sample] = self.load_data(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        embedding = sample.embedding
        label = sample.label.value
        return embedding, label

    def load_data(self, path: str) -> List[Sample]:
        with open(path, "rb") as f:
            return pickle.load(f)
