import os
import sys
import argparse
import json
import urllib.parse
import logging
from typing import List, Dict, Tuple, Optional
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np

from pero_ocr.document_ocr.layout import PageLayout, TextLine

from textbite.utils import LineLabel


Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")


@dataclass
class AnnotatedRegion:
    id: str
    label: str
    bbox: Rectangle
    lines: List[str] = field(default_factory=list)


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True, type=str, help="Path to an exported JSON file from label-studio.")
    parser.add_argument('--xml', required=True, type=str, help="Path to a folder containing XML files from PERO-OCR.")

    args = parser.parse_args()
    return args


def parse_annotated_file(annotated_file: dict) -> Tuple[List[AnnotatedRegion], Dict[str, List[str]]]:
    annotations = annotated_file["annotations"]

    if len(annotations) > 1:
        logging.warning(f"Multiple annotations present for id: {annotated_file['id']}")

    annotations = annotations[0]["result"]

    # Relations
    relations_list = [a for a in annotations if a["type"] == "relation"]
    relations = {r["from_id"]: [] for r in relations_list}
    for r in relations_list:
        relations[r["from_id"]].append(r["to_id"])

    # Regions
    regions = [a for a in annotations if a["type"] == "rectanglelabels"]
    regions = [parse_annotated_region(region) for region in regions]
    
    return regions, relations


def parse_annotated_region(region: dict) -> AnnotatedRegion:
    id = region["id"]
    label = region["value"]["rectanglelabels"][0]
    bbox = get_region_bbox(region)

    return AnnotatedRegion(id=id, label=label, bbox=bbox)


def get_region_bbox(region: dict) -> Rectangle:
    im_width = region["original_width"]
    im_height = region["original_height"]

    region_width = (region["value"]["width"] / 100.0) * im_width
    region_height = (region["value"]["height"] / 100.0) * im_height

    xmin = (region["value"]["x"] / 100.0) * im_width
    ymin = (region["value"]["y"] / 100.0) * im_height
    xmax = xmin + region_width
    ymax = ymin + region_height

    return Rectangle(xmin, ymin, xmax, ymax)


def get_polygon_bbox(polygon: np.ndarray) -> Rectangle:
    mins = np.min(polygon, axis=0)
    maxs = np.max(polygon, axis=0)

    # (minx, miny, maxx, maxy)
    return Rectangle(mins[0], mins[1], maxs[0], maxs[1])


def bbox_intersection(lhs: Rectangle, rhs: Rectangle) -> float:
    dx = min(lhs.xmax, rhs.xmax) - max(lhs.xmin, rhs.xmin)
    dy = min(lhs.ymax, rhs.ymax) - max(lhs.ymin, rhs.ymin)

    return dx * dy if dx >= 0 and dy >= 0 else 0.0


def create_mapping(
        regions: List[AnnotatedRegion],
        pagexml: PageLayout,
    ) -> None:
        for line in pagexml.lines_iterator():
            if line.transcription:
                map_line(line, regions)


def map_line(line: TextLine, regions: List[AnnotatedRegion]) -> None:
    line_bbox: Rectangle = get_polygon_bbox(line.polygon)
    candidates = []

    best_region = None
    best_intersection = 0.0
    for region in regions:
        intersection = bbox_intersection(line_bbox, region.bbox)
        if intersection > best_intersection:
            best_intersection = intersection
            best_region = region

    if best_region:
        best_region.lines.append(line.transcription)


def add_labels(regions: List[AnnotatedRegion], relations: Dict[str, List[str]]):
    for region in regions:
        if region.label == "title":
            for line in region.lines:
                print(f"{line}\t{LineLabel.TITLE.value}")
        
        elif region.label == "text":
            for line in region.lines[:-1]:
                print(f"{line}\t{LineLabel.NONE.value}")

            label = LineLabel.NONE if region.id in relations else LineLabel.TERMINATING
            print(f"{region.lines[-1]}\t{label.value}")
        else:
            logging.warning(f"Unknown region: {region.label}")

        print()


def main(args):
    with open (args.json, "r") as f:
        annotations_file_json = json.load(f)

    for annotated_file in annotations_file_json:
        filename = os.path.basename(annotated_file["data"]["image"])[:-4]
        filename = urllib.parse.unquote(filename)
        filename_xml = f"{filename}.xml"

        path_xml = os.path.join(args.xml, filename_xml)
        try:
            pagexml = PageLayout(file=path_xml)
        except OSError:
            logging.warning(f"XML file {path_xml} not found. SKIPPING")
            continue

        regions, relations = parse_annotated_file(annotated_file)
        create_mapping(regions, pagexml)
        regions = [region for region in regions if region.lines]
        add_labels(regions, relations)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
