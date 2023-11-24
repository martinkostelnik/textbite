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

    parser.add_argument("--data", required=True, type=str, help="Path to a folder with json, xml and jpg data.")

    args = parser.parse_args()
    return args


def draw_bites(img):
    pass


def main(args):
    image_files = [image_file for image_file in os.listdir(args.data) if image_file.endswith(".jpg")]
    for img_filename in image_files:
        pagexml_path = os.path.join(args.data, img_filename.replace(".jpg", ".xml"))
        pagexml = PageLayout(file=pagexml_path)

        regions_path = os.path.join(args.data, img_filename.replace(".jpg", ".json"))
        with open(regions_path, "r") as f:
            bites = json.load(f)

        img_path = os.path.join(args.data, img_filename)
        img = cv2.imread(img_path)

        overlay = np.zeros_like(img)

        for i, bite in enumerate(bites):
            for line_id in bite:
                for line in pagexml.lines_iterator():
                    if line.id.strip() == line_id:
                        mask = np.zeros_like(img)
                        pts = line.polygon
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(mask, [pts], colors[i])
                        overlay = cv2.addWeighted(overlay, 1, mask, 1-ALPHA, 0)
                        break

        result = cv2.addWeighted(img, 1, overlay, 1-ALPHA, 0)
        res_filename = os.path.join(args.data, img_filename.replace(".jpg", "-vis.jpg"))
        cv2.imwrite(res_filename, result)


if __name__ == "__main__":  
    logging.basicConfig(
        level=logging.DEBUG,
        force=True,
    )
    args = parse_arguments()
    main(args)
