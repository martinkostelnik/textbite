"""Visualize page geometry created from pagexml into an jpg.

    Date -- 02.12.2023
    Author -- Martin Kostelnik
"""

import argparse
import sys
import os
import logging

import cv2

from textbite.geometry import PageGeometry


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--split", action="store_true", help="Split geometry into cohesive pieces.")
    parser.add_argument("--save", required=True, type=str, help="Path to the output folder.")

    args = parser.parse_args()
    return args


def render_geometry(image, geometry: PageGeometry):
    for line in geometry.lines:
        if line.parent: # PARENT RED
            start = (int(line.center.x) + 20, int(line.center.y))
            end = (int(line.parent.center.x) + 20, int(line.parent.center.y))
            cv2.line(image, start, end, (0, 0, 255), 7)

        if line.child: # CHILD GREEN
            start = (int(line.center.x), int(line.center.y))
            end = (int(line.child.center.x), int(line.child.center.y))
            cv2.line(image, start, end, (0, 255, 0), 7)

    return image


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    os.makedirs(args.save, exist_ok=True)
    image_filenames = [filename for filename in os.listdir(args.images) if filename.endswith(".jpg")]

    for image_filename in image_filenames:
        image_path = os.path.join(args.images, image_filename)
        xml_filename = image_filename.replace(".jpg", ".xml")
        xml_path = os.path.join(args.xml, xml_filename)

        img = cv2.imread(image_path)
        try:
            geometry = PageGeometry(path=xml_path)
        except OSError:
            logging.warning(f"XML file corresponding to image {image_filename} not found. Skipping.")
            continue
        
        if args.split:
            geometry.split_geometry()

        img = render_geometry(img, geometry)
        save_path = os.path.join(args.save, image_filename.replace(".jpg", "-geometry.jpg"))
        cv2.imwrite(save_path, img)

        logging.info(f'Processed {image_filename}')


if __name__ == "__main__":
    main()
