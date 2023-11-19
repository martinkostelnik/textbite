import os
import argparse
import sys
from typing import Tuple, List


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", type=str, help="Path to a folder containing ground truths.")
    parser.add_argument("--pred", type=str, help="Path to a folder containing predictions.",)

    args = parser.parse_args()
    return args


def compare_reading_order(gt: List[str], pred: List[str]) -> Tuple[int, int]:
    position_hits = sum([true_region.strip() == pred_region.strip() for true_region, pred_region in zip(gt, pred)])
    continuity_hits = 0

    for idx, true_region in enumerate(gt[:-1]):
        next_true_region = gt[idx + 1].strip()
        try:
            next_pred_region = pred[pred.index(true_region) + 1].strip()
        except IndexError:
            continue

        continuity_hits += (next_true_region == next_pred_region)

    return position_hits, continuity_hits


def main(args):
    regions = 0
    position_hits = 0
    continuity_hits = 0
    files = 0

    for filename in os.listdir(args.pred):
        if not filename.endswith(".txt"):
            continue
        files += 1

        with open(os.path.join(args.pred, filename), "r") as f:
            pred = f.read().strip().split("\n\n")

        with open(os.path.join(args.gt, filename), "r") as f:
            gt = f.read().strip().split("\n\n")

        if len(pred) != len(gt):
            print(f"WARNING: {filename} | pred: {len(pred)} | gt: {len(gt)}")
            print(set(pred) - set(gt))
            continue

        p, c = compare_reading_order(gt, pred)
        regions += len(pred)
        position_hits += p
        continuity_hits += c

    position_accuracy = position_hits / regions * 100
    continuity_accuracy = continuity_hits / (regions - files) * 100

    print(f"Compared a total of {files} files with a total of {regions} regions.")
    print(f"Region position accuracy:\t{position_accuracy:.2f} % ({position_hits}/{regions})")
    print(f"Region continuity accuracy:\t{continuity_accuracy:.2f} % ({continuity_hits}/{regions - files})")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
