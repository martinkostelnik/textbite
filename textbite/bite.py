from dataclasses import dataclass, field
from typing import List
import json

from textbite.geometry import AABB


@dataclass
class Bite:
    cls: str
    bbox: AABB
    lines: List[str] = field(default_factory=list)
    name: str = ""


def load_bites(path: str) -> List[Bite]:
    with open(path, "r") as f:
        return json.load(
            f,
            object_hook=lambda d: Bite(
                d["cls"],
                AABB(d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]),
                d["lines"],
                d["name"]
            )
        )
