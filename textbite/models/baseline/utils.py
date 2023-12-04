from dataclasses import dataclass

from torch import FloatTensor

from textbite.utils import LineLabel


@dataclass
class Sample:
    embedding: FloatTensor
    label: LineLabel

    def __iter__(self):
        return iter((self.embedding, self.label))

