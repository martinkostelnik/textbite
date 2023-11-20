from enum import Enum
from typing import List, Tuple

import segeval


CZERT_PATH = r"UWB-AIR/Czert-B-base-cased"


class LineLabel(Enum):
    NONE = 0
    TERMINATING = 1
    TITLE = 2


#  .....REG1.....  .....REG2.....  .....REG3.....
# [[line1, line2], [line3, line4], [line5, line6]]  | raw
#     1      1        2      2        3      3      | positions
#        2               2               2          | masses
def segmentation_metrics(gt: List[List[str]], pred: List[List[str]]) -> Tuple[float, float]:
    gt_positions = [len(region) for region in gt]
    pred_positions = [len(region) for region in pred]
    
    boundary_similarity = segeval.boundary_similarity(gt_positions, pred_positions)
    segmentation_similarity = segeval.segmentation_similarity(gt_positions, pred_positions)

    return boundary_similarity, segmentation_similarity


if __name__ == "__main__":
    gt = [["Ahoj,", "jmenuju se Martin"], ["Čus, ja jsem Petr."], ["A ja", "jsem Oldrich"]]
    pred = [["Ahoj,", "jmenuju se Martin"], ["Čus, ja jsem Petr."], ["A ja"], ["jsem Oldrich"]]
    segmentation_metrics(gt, pred)
    segmentation_metrics(gt, gt)
