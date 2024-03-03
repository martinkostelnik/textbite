"""Visualize visibility of couple of random lines in page geometry.

    Date -- 06.12.2023
    Author -- Martin Kostelnik
"""

import argparse
import sys
import os
import logging
import random

import numpy as np
import cv2
from ultralytics import YOLO

from pero_ocr.document_ocr.layout import PageLayout

from textbite.geometry import PageGeometry
from textbite.models.yolo.infer import YoloBiter


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images.")
    parser.add_argument("--model", required=True, type=str, help="Path to pretrained YOLO model.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--save", required=True, type=str, help="Path to the output folder.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    biter = YoloBiter(YOLO(args.model))

    os.makedirs(args.save, exist_ok=True)
    image_filenames = [filename for filename in os.listdir(args.images) if filename.endswith(".jpg")]

    for image_filename in image_filenames:
        image_path = os.path.join(args.images, image_filename)
        xml_filename = image_filename.replace(".jpg", ".xml")
        xml_path = os.path.join(args.xml, xml_filename)

        pagexml = PageLayout(file=xml_path)
        bites = biter.produce_bites(image_path, pagexml)
        regions = [bite.bbox for bite in bites]

        img = cv2.imread(image_path)
        try:
            geometry = PageGeometry(path=xml_path, regions=regions)
        except OSError:
            logging.warning(f"XML file corresponding to image {image_filename} not found. Skipping.")
            continue
        
        geometry.set_region_visibility()

        img = cv2.imread(image_path)

        random_region_idxs = []
        while len(random_region_idxs) != 1:
            r = random.randint(0, len(geometry.regions)-1)
            if r not in random_region_idxs:
                random_region_idxs.append(r)

        overlay = np.zeros_like(img)
        for random_region_idx in random_region_idxs:
            random_region = geometry.regions[random_region_idx]

            for ve in random_region.visible_entities:
                cv2.rectangle(img, (int(ve.bbox.xmin), int(ve.bbox.ymin)), (int(ve.bbox.xmax), int(ve.bbox.ymax)), (255, 0, 0), 6)

            xmin = int(random_region.bbox.xmin)
            ymin = int(random_region.bbox.ymin)
            xmax = int(random_region.bbox.xmax)
            ymax = int(random_region.bbox.ymax)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 6)

        save_path = os.path.join(args.save, image_filename.replace(".jpg", "-neighbours.jpg"))
        cv2.imwrite(save_path, img)

        logging.info(f'Processed {image_filename}')


if __name__ == "__main__":
    main()
