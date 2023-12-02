from enum import Enum
from typing import List, Dict


CZERT_PATH = r"UWB-AIR/Czert-B-base-cased"


class LineLabel(Enum):
    NONE = 0
    TERMINATING = 1
    TITLE = 2


def get_line_clusters(bites: List[List[str]]) -> Dict[str, int]:
    return {line_id: bite_id for bite_id, bite in enumerate(bites) for line_id in bite}
