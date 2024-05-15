"""Visualize page geometry created from pagexml into an jpg.

    Date -- 02.12.2023
    Author -- Martin Kostelnik
"""


import argparse
import sys
import os
import logging

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


def render_geometry(image, geometry: PageGeometry):
    h, w, _ = image.shape

    for region in geometry.regions:
        if region.parent: # PARENT RED
            start = (int(region.center.x) + 20, int(region.center.y))
            end = (int(region.parent.center.x) + 20, int(region.parent.center.y))
            cv2.line(image, start, end, (0, 0, 255), 7)

        if region.child: # CHILD GREEN
            start = (int(region.center.x), int(region.center.y))
            end = (int(region.child.center.x), int(region.child.center.y))
            cv2.line(image, start, end, (0, 255, 0), 7)

        # Draw center point
        x = int(region.center.x * w)
        y = int(region.center.x * h)
        cv2.circle(image, (int(region.center.x), int(region.center.y)), 15, (255, 0, 0), 10)

    return image


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
            geometry = PageGeometry(regions=regions, path=xml_path)
        except OSError:
            logging.warning(f"XML file corresponding to image {image_filename} not found. Skipping.")
            continue
        
        img = render_geometry(img, geometry)
        save_path = os.path.join(args.save, image_filename.replace(".jpg", "-geometry.jpg"))
        cv2.imwrite(save_path, img)

        logging.info(f'Processed {image_filename}')


if __name__ == "__main__":
    main()
