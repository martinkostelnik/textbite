"""Visualizing the effect of the GNN compared to YOLO

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


import logging
import sys
import argparse
import os
import json
from typing import List

import cv2
from cv2.typing import MatLike
import numpy as np

from pero_ocr.core.layout import PageLayout

from textbite.visualization.utils import COLORS, overlay_line
from textbite.utils import get_line_clusters
from textbite.bite import load_bites, Bite


ALPHA = 0.3


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with jpg data.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--original", required=True, type=str, help="Path to a folder with original json data.")
    parser.add_argument("--joiner", required=True, type=str, help="Path to a folder with joined json data.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put the outputs")

    args = parser.parse_args()
    return args


def draw_bites(img: MatLike, pagexml: PageLayout, bites: List[Bite]) -> MatLike:
    overlay = np.zeros_like(img)

    for line in pagexml.lines_iterator():
        if not line.transcription.strip():
            continue
        
        line_found = False
        for bite_idx, bite in enumerate(bites):
            if line.id in bite.lines:
                line_found = True
                overlay = overlay_line(overlay, line, COLORS[bite_idx % len(COLORS)], ALPHA)

        if not line_found:
            logging.warning(f"Line {line.id} with transcription {repr(line.transcription)} not found in any bite.")
            continue

    return cv2.addWeighted(img, 1, overlay, 1-ALPHA, 0)


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    os.makedirs(args.save, exist_ok=True)

    json_filenames = [json_filename for json_filename in os.listdir(args.joiner) if json_filename.endswith(".json")]

    for json_filename in json_filenames:
        path_json_joiner = os.path.join(args.joiner, json_filename)
        path_json_orig = os.path.join(args.original, json_filename)
        path_xml = os.path.join(args.xml, json_filename.replace(".json", ".xml"))
        path_img = os.path.join(args.images, json_filename.replace(".json", ".jpg"))

        try:
            pagexml = PageLayout(file=path_xml)
        except OSError:
            logging.warning(f"XML {path_xml} not found, skipping.")
            continue

        try:
            img = cv2.imread(path_img)
        except OSError:
            logging.warning(f"Image {path_img} not found, skipping.")
            continue

        original_bites = load_bites(path_json_orig)
        joiner_bites = load_bites(path_json_joiner)

        result_orig = draw_bites(img, pagexml, original_bites)
        result_joiner = draw_bites(img, pagexml, joiner_bites)

        combined_width = result_orig.shape[1] * 2
        result = np.zeros((result_orig.shape[0], combined_width, 3), dtype=np.uint8)

        # Paste the first image onto the blank image at the appropriate position
        result[:, :result_orig.shape[1]] = result_orig
        # Paste the second image next to the first one
        result[:, result_joiner.shape[1]:] = result_joiner

        res_filename = os.path.join(args.save, json_filename.replace(".json", "-bites.jpg"))
        cv2.imwrite(res_filename, result)

        logging.info(f'Processed {json_filename}')


if __name__ == "__main__":
    main()
