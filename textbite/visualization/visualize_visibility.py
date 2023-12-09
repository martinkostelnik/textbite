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

from textbite.geometry import PageGeometry
from textbite.visualization.utils import overlay_line


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--save", required=True, type=str, help="Path to the output folder.")

    args = parser.parse_args()
    return args


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
        
        geometry.set_visibility()

        img = cv2.imread(image_path)

        random_line_idxs = []
        while len(random_line_idxs) != 2:
            r = random.randint(0, len(geometry.lines)-1)
            if r not in random_line_idxs:
                random_line_idxs.append(r)

        overlay = np.zeros_like(img)
        for random_line_idx in random_line_idxs:
            random_line = geometry.lines[random_line_idx]

            for l in random_line.visible_lines:
                overlay = overlay_line(overlay, l.text_line, (255, 0, 0), 0.3)

            overlay = overlay_line(overlay, random_line.text_line, (0, 255, 0), 0.3)

        img = cv2.addWeighted(img, 1, overlay, 1-0.3, 0)

        save_path = os.path.join(args.save, image_filename.replace(".jpg", "-neighbours.jpg"))
        cv2.imwrite(save_path, img)

        logging.info(f'Processed {image_filename}')


if __name__ == "__main__":
    main()
