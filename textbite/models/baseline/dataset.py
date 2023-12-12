from typing import List
import pickle

import torch

from textbite.embedding import LineEmbedding


class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[LineEmbedding]):
        self.data: List[LineEmbedding] = data
        self.data = [sample for sample in self.data if sample.label is not None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        embedding = sample.embedding
        label = sample.label.value
        return embedding, label
