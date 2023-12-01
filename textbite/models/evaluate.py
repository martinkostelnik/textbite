#!/usr/bin/env python3
import argparse
import json
import logging
import os

import sklearn.metrics

from textbite.annotation_manipulation import get_line_clusters


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-page', action='store_true',
                        help='Treat the paths as single files.')
    parser.add_argument('-s', '--hypothesis', required=True,
                        help='Path to system outputs')
    parser.add_argument('-g', '--ground-truth', required=True,
                        help='Path to ground truth')

    return parser.parse_args()


def score_page(hypothesis, ground_truth):
    line_ids = sum(hypothesis, []) + sum(ground_truth, [])

    hyp_lines_clusters = get_line_clusters(hypothesis)
    gt_lines_clusters = get_line_clusters(ground_truth)

    hyp = [hyp_lines_clusters[line_id] for line_id in line_ids]
    gt = [gt_lines_clusters[line_id] for line_id in line_ids]

    return sklearn.metrics.homogeneity_completeness_v_measure(hyp, gt)


def main():
    args = get_args()

    if args.single_page:
        with open(args.hypothesis, "r") as f:
            hypothesis = json.load(f)

        with open(args.ground_truth, "r") as f:
            ground_truth = json.load(f)

        score = score_page(hypothesis, ground_truth)
        print(score)

    else:
        nb_not_found = 0
        nb_found = 0
        accumulated_score = None
        for gt_fn in [fn for fn in os.listdir(args.ground_truth) if fn.endswith(".json")]:
            with open(os.path.join(args.ground_truth, gt_fn)) as f:
                ground_truth = json.load(f)

            try:
                with open(os.path.join(args.hypothesis, gt_fn)) as f:
                    hypothesis = json.load(f)
            except FileNotFoundError:
                nb_not_found += 1
                continue

            score = score_page(hypothesis, ground_truth)
            nb_found += 1
            if not accumulated_score:
                accumulated_score = score
            else:
                accumulated_score = [a + s for a, s in zip(accumulated_score, score)]

            print(gt_fn, score)

        accumulated_score = [a / nb_found for a in accumulated_score]
        print("complete", accumulated_score)

        logging.warning(f'Results partial, {nb_not_found} pages from ground truth were not matched')


if __name__ == '__main__':
    main()
