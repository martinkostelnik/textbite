from typing import List
import pickle

import numpy as np
import torch
from scipy.spatial import distance

from textbite.models.baseline.utils import Sample


class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.data: List[Sample] = self.load_data(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample_embedding, sample_label = self.data[index]
        left_context2_embedding = torch.zeros_like(sample_embedding)
        left_context1_embedding = torch.zeros_like(sample_embedding)
        right_context1_embedding = torch.zeros_like(sample_embedding)
        right_context2_embedding = torch.zeros_like(sample_embedding)

        d = [-10.0] * 4

        if index > 1:
            left_context2_embedding, _ = self.data[index - 2]
            d[0] = np.linalg.norm(sample_embedding - left_context2_embedding)
            # d[0] = distance.cosine(sample_embedding, left_context2_embedding)

        if index > 0:
            left_context1_embedding, _ = self.data[index - 1]
            d[1] = np.linalg.norm(sample_embedding - left_context1_embedding)
            # d[1] = distance.cosine(sample_embedding, left_context1_embedding)

        if index < len(self) - 1:
            right_context1_embedding, _ = self.data[index + 1]
            d[3] = np.linalg.norm(sample_embedding - right_context1_embedding)
            # d[2] = distance.cosine(sample_embedding, right_context1_embedding)

        if index < len(self) - 2:
            right_context2_embedding, _ = self.data[index + 2]
            d[3] = np.linalg.norm(sample_embedding - right_context2_embedding)
            # d[3] = distance.cosine(sample_embedding, right_context2_embedding)

        # embedding = torch.cat((left_context1_embedding, sample_embedding, right_context1_embedding))
        # embedding = torch.tensor(d, dtype=torch.float32)
        embedding = sample_embedding

        return embedding, sample_label.value

    def load_data(self, path: str) -> List[Sample]:
        with open(path, "rb") as f:
            return pickle.load(f)
