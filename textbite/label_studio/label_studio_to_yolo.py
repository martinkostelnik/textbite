import os
import sys
import argparse
import json
import urllib.parse
import logging
from typing import List, Dict, Tuple
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np

from pero_ocr.document_ocr.layout import PageLayout, TextLine


Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")


@dataclass
class AnnotatedRegion:
    id: str
    label: str
    bbox: Rectangle
    imsize: Tuple[int, int]
    yolo: str = field(default_factory=str)
    lines: List[TextLine] = field(default_factory=list)


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True, type=str, help="Path to an exported JSON file from label-studio.")
    parser.add_argument('--xml', required=True, type=str, help="Path to a folder containing XML files from PERO-OCR.")
    parser.add_argument("--save", required=True, type=str, help="Path to a folder where results will be saved.")

    args = parser.parse_args()
    return args


def parse_annotated_file(annotated_file: dict) -> Tuple[List[AnnotatedRegion], Dict[str, List[str]]]:
    annotations = annotated_file["annotations"]

    if len(annotations) > 1:
        logging.warning(f"Multiple annotations present for id: {annotated_file['id']}")

    annotations = annotations[0]["result"]

    # Regions
    regions = [a for a in annotations if a["type"] == "rectanglelabels"]
    regions = [parse_annotated_region(region) for region in regions]
    
    return regions


def parse_annotated_region(region: dict) -> AnnotatedRegion:
    id = region["id"]
    label = region["value"]["rectanglelabels"][0]
    bbox = get_region_bbox(region)
    # yolo = get_region_yolo(region)
    
    im_width = region["original_width"]
    im_height = region["original_height"]
    imsize = (im_width, im_height)

    return AnnotatedRegion(id=id, label=label, bbox=bbox, imsize=imsize)


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


def get_region_yolo(region: dict) -> str:
    x = (region["value"]["x"] + (region["value"]["width"] / 2)) / 100.0
    y = (region["value"]["y"] + (region["value"]["height"] / 2)) / 100.0
    width = region["value"]["width"] / 100.0
    height = region["value"]["height"] / 100.0

    return f"{x} {y} {width} {height}"


def get_region_lines_yolo(lines: List[TextLine], imsize: Tuple[int, int]) -> str:
    bboxes = [get_polygon_bbox(line.polygon) for line in lines]

    min_x = min(bboxes, key=lambda x: x.xmin).xmin
    min_y = min(bboxes, key=lambda x: x.ymin).ymin
    max_x = max(bboxes, key=lambda x: x.xmax).xmax
    max_y = max(bboxes, key=lambda x: x.ymax).ymax
    bbox = Rectangle(min_x, min_y, max_x, max_y)

    return get_line_yolo(bbox, imsize)


def get_line_yolo(bbox: Rectangle, imsize: Tuple[int, int]) -> str:
    x = (bbox.xmin + ((bbox.xmax - bbox.xmin) / 2)) / imsize[0]
    y = (bbox.ymin + ((bbox.ymax - bbox.ymin) / 2)) / imsize[1]
    width = (bbox.xmax - bbox.xmin) / imsize[0]
    height = (bbox.ymax - bbox.ymin) / imsize[1]

    return f"{x} {y} {width} {height}"


def get_line_y(line: TextLine) -> (float, float):
    bbox = get_polygon_bbox(line.polygon)

    y = bbox.ymin + ((bbox.ymax - bbox.ymin) / 2.0)

    return y


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

    best_region = None
    best_intersection = 0.0
    for region in regions:
        intersection = bbox_intersection(line_bbox, region.bbox)
        if intersection > best_intersection:
            best_intersection = intersection
            best_region = region

    if best_region:
        best_region.lines.append(line)


def get_result_str(regions: List[AnnotatedRegion]) -> str:
    result_str = ""
    for region in regions:
        if region.label == "text":
            sorted_lines = sorted(region.lines, key=lambda x: get_line_y(x))
            first = sorted_lines[0]
            last = sorted_lines[-1]

            first_bbox = get_polygon_bbox(first.polygon)
            last_bbox = get_polygon_bbox(last.polygon)

            result_str += f"0 {get_region_lines_yolo(region.lines, region.imsize)}\n"
            result_str += f"1 {get_line_yolo(first_bbox, region.imsize)}\n"
            result_str += f"2 {get_line_yolo(last_bbox, region.imsize)}\n"
        elif region.label == "title":
            result_str += f"3 {get_region_lines_yolo(region.lines, region.imsize)}\n"
        else:
            continue

    return result_str


def save_result(path, result_str):
    with open(path, "w") as f:
        print(result_str, file=f, end="")


def main(args):
    with open (args.json, "r") as f:
        annotations_file_json = json.load(f)

    os.makedirs(args.save, exist_ok=True)

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

        regions = parse_annotated_file(annotated_file)
        create_mapping(regions, pagexml)
        regions = [region for region in regions if region.lines]

        result_str = get_result_str(regions)

        filename_yolo = f"{filename}.yolo"
        save_path = os.path.join(args.save, filename_yolo)
        save_result(save_path, result_str)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
