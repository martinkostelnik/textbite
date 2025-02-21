"""Visualize graphs for joiner model.

    Date -- 02.03.2024
    Author -- Martin Kostelnik
"""


import argparse
import sys
import os
import logging
from typing import List

import cv2
from ultralytics import YOLO

from pero_ocr.core.layout import PageLayout

from textbite.geometry import AABB, bbox_center
from textbite.models.yolo.infer import YoloBiter
from textbite.models.joiner.graph import JoinerGraphProvider, Graph
from textbite.data_processing.label_studio import LabelStudioExport


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images.")
    parser.add_argument("--model", required=True, type=str, help="Path to pretrained YOLO model.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--export", required=True, type=str, help="Path to label studio export.")
    parser.add_argument("--save", required=True, type=str, help="Path to the output folder.")

    args = parser.parse_args()
    return args


def render_graph(image, graph: Graph, regions: List[AABB]):
    h, w, _ = image.shape

    i = 0
    for idx, region in enumerate(regions):
        center = bbox_center(region)
        center_x = int(center.x)
        center_y = int(center.y)
        xmin = int(region.xmin)
        ymin = int(region.ymin)
        xmax = int(region.xmax)
        ymax = int(region.ymax)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 6)

        start = (center_x, center_y)
        for label, (from_idx, to_idx) in zip(graph.labels, graph.edge_index.T):
            label = label.item()
            if label != 1 or from_idx != idx:
                continue
            to_idx = to_idx.item()
            end_region = regions[to_idx]
            to_center = bbox_center(end_region)
            end_center_x = int(to_center.x)
            end_center_y = int(to_center.y)
            end = (end_center_x, end_center_y)
            cv2.line(image, start, end, (0, 0, 255), 7)
            i += 1

    return image, i


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    biter = YoloBiter(YOLO(args.model))
    graph_provider = JoinerGraphProvider()
    export = LabelStudioExport(path=args.export)

    os.makedirs(args.save, exist_ok=True)
    image_filenames = [filename for filename in os.listdir(args.images) if filename.endswith(".jpg")]

    for image_filename in image_filenames:
        image_path = os.path.join(args.images, image_filename)
        xml_filename = image_filename.replace(".jpg", ".xml")
        xml_path = os.path.join(args.xml, xml_filename)

        try:
            pagexml = PageLayout(file=xml_path)
        except OSError:
            logging.warning(f"XML file corresponding to image {image_filename} not found. Skipping.")
            continue

        img = cv2.imread(image_path)
        doc = export.documents_dict[image_filename]

        try:
            graph, regions = graph_provider.get_graph_from_file(biter, image_path, pagexml, doc)
        except RuntimeError:
            continue
        
        img, i = render_graph(img, graph, regions)
        save_path = os.path.join(args.save, f'{i}-{image_filename.replace(".jpg", "-geometry.jpg")}')
        cv2.imwrite(save_path, img)

        logging.info(f'Processed {image_filename}')


if __name__ == "__main__":
    main()