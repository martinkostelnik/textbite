import logging
import sys
import argparse
import os
import json

import cv2
import numpy as np

from pero_ocr.document_ocr.layout import PageLayout


ALPHA = 0.3
colors = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (34, 139, 34),    # Forest Green
    (70, 130, 180),   # Steel Blue
    (255, 20, 147),   # Deep Pink
    (218, 112, 214),  # Orchid
    (255, 165, 0),    # Orange
    (173, 216, 230),  # Light Blue
    (255, 69, 0),     # Red-Orange
    (0, 191, 255),    # Deep Sky Blue
    (128, 0, 128),    # Purple
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 99, 71),    # Tomato
    (255, 192, 203),  # Pink
    (32, 178, 170),   # Light Sea Green
    (250, 128, 114),  # Salmon
    (0, 128, 128),    # Teal
    (240, 230, 140)   # Khaki
]


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--pages", required=True, type=str, help="Path to a folder with xml and jpg data.")
    parser.add_argument("--jsons", required=True, type=str, help="Path to a folder with json data.")
    parser.add_argument("--out-dir", required=True, type=str, help="Folder where to put the outputs")

    args = parser.parse_args()
    return args


def draw_bites(img, pagexml, bites):
    overlay = np.zeros_like(img)

    rev_bites_dict = {line_id: bite_id for bite_id, bite in enumerate(bites) for line_id in bite}

    for line in pagexml.lines_iterator():
        mask = np.zeros_like(img)
        pts = line.polygon
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], colors[rev_bites_dict[line.id] % len(colors)])
        overlay = cv2.addWeighted(overlay, 1, mask, 1-ALPHA, 0)

    return cv2.addWeighted(img, 1, overlay, 1-ALPHA, 0)


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    os.makedirs(args.out_dir, exist_ok=True)

    image_files = [image_file for image_file in os.listdir(args.pages) if image_file.endswith(".jpg")]
    for img_filename in image_files:

        pagexml_path = os.path.join(args.pages, img_filename.replace(".jpg", ".xml"))
        pagexml = PageLayout(file=pagexml_path)

        regions_path = os.path.join(args.jsons, img_filename.replace(".jpg", ".json"))
        with open(regions_path, "r") as f:
            bites = json.load(f)

        img_path = os.path.join(args.pages, img_filename)
        img = cv2.imread(img_path)

        result = draw_bites(img, pagexml, bites)

        res_filename = os.path.join(args.out_dir, img_filename.replace(".jpg", "-vis.jpg"))
        cv2.imwrite(res_filename, result)

        logging.info(f'Processed {res_filename}')


if __name__ == "__main__":
    main()
