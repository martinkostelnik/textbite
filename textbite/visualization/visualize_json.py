"""Visualize page bites created by a model or hand annotated from json containing
   results and xml file into an jpg.

    Date -- 02.12.2023
    Author -- Martin Kostelnik, Karel Benes
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

from pero_ocr.document_ocr.layout import PageLayout

from textbite.visualization.utils import COLORS, overlay_line
from textbite.utils import get_line_clusters


ALPHA = 0.3


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with jpg data.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--jsons", required=True, type=str, help="Path to a folder with json data.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put the outputs")

    args = parser.parse_args()
    return args


def draw_bites(img: MatLike, pagexml: PageLayout, bites: List[List[str]]) -> MatLike:
    overlay = np.zeros_like(img)

    rev_bites_dict = get_line_clusters(bites)

    for line in pagexml.lines_iterator():
        if line.id not in rev_bites_dict:
            logging.warning(f'Line {line.id} not in any bite of {pagexml.id}')
            continue

        overlay = overlay_line(overlay, line, COLORS[rev_bites_dict[line.id] % len(COLORS)], ALPHA)

    return cv2.addWeighted(img, 1, overlay, 1-ALPHA, 0)


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    os.makedirs(args.save, exist_ok=True)

    json_filenames = [json_filename for json_filename in os.listdir(args.jsons) if json_filename.endswith(".json")]

    for json_filename in json_filenames:
        path_json = os.path.join(args.jsons, json_filename)
        with open(path_json, "r") as f:
            bites = json.load(f)

        path_xml = os.path.join(args.xml, json_filename.replace(".json", ".xml"))
        pagexml = PageLayout(file=path_xml)

        path_img = os.path.join(args.images, json_filename.replace(".json", ".jpg"))
        img = cv2.imread(path_img)
        if img is None:
            continue

        result = draw_bites(img, pagexml, bites)

        res_filename = os.path.join(args.save, json_filename.replace(".json", "-bites.jpg"))
        cv2.imwrite(res_filename, result)

        logging.info(f'Processed {json_filename}')


if __name__ == "__main__":
    main()
