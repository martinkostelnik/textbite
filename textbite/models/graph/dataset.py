from typing import List
import pickle

import torch

from textbite.models.graph.create_graphs import Graph 


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.data: List[Graph] = self.load_data(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        return sample.node_features, sample.edge_index, sample.labels

    def load_data(self, path: str) -> List[Graph]:
        with open(path, "rb") as f:
            return pickle.load(f)
