from collections import namedtuple
import os
import sys
import argparse
from typing import List, Tuple
import pickle
from math import sqrt

import numpy as np
import cv2
import torch
from transformers import BertModel, BertTokenizerFast
from safe_gpu import safe_gpu

from pero_ocr.document_ocr.layout import PageLayout, TextLine

from semant.language_modelling.model import build_model
from semant.language_modelling.tokenizer import build_tokenizer

from textbite.models.baseline.utils import Sample, LineLabel
from textbite.utils import CZERT_PATH
from textbite.models.baseline.create_embeddings import get_embedding


Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--mapping", required=True, type=str, help="Path to a mapping file.")
    parser.add_argument("--model", type=str, help="Path to a custom BERT model.")
    parser.add_argument("--save", required=True, type=str, help="Path to the output folder.")

    args = parser.parse_args()
    return args


class Line:
    Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

    def __init__(self, text_line, region):
        self.text_line = text_line
        self.region = region
        self.parent = None
        self.child = None
        
        self.set_geometry()


    def set_geometry(self) -> None:
        mins = np.min(self.text_line.polygon, axis=0)
        maxs = np.max(self.text_line.polygon, axis=0)

        # (minx, miny, maxx, maxy)
        self.bbox = Line.Rectangle(mins[0], mins[1], maxs[0], maxs[1])

        x = self.bbox.xmin + ((self.bbox.xmax - self.bbox.xmin) / 2.0)
        y = self.bbox.ymin + ((self.bbox.ymax - self.bbox.ymin) / 2.0)

        self.center = (x, y)

    def set_parent(self, lines):
        y = self.center[1]
        parent_candidates = [line for line in lines if line.center[1] < y]
        parent_candidates = [line for line in parent_candidates if self.x_candidate_predicate(line)]
        if parent_candidates:
            self.parent = sorted(parent_candidates, reverse=True, key=lambda x: x.center[1])[0]

    def set_child(self, lines):
        y = self.center[1]
        child_candidates = [line for line in lines if line.center[1] > y]
        child_candidates = [line for line in child_candidates if self.x_candidate_predicate(line)]
        if child_candidates:
            self.child = sorted(child_candidates, key=lambda x: x.center[1])[0]

    def x_candidate_predicate(self, ref) -> bool:
        return    (self.bbox.xmin >= ref.bbox.xmin and self.bbox.xmin <= ref.bbox.xmax) \
               or (self.bbox.xmax >= ref.bbox.xmin and self.bbox.xmax <= ref.bbox.xmax) \
               or (ref.bbox.xmin >= self.bbox.xmin and ref.bbox.xmin <= self.bbox.xmax) \
               or (ref.bbox.xmax >= self.bbox.xmin and ref.bbox.xmax <= self.bbox.xmax)

    def children_iterator(self):
        ptr = self.child
        while ptr:
            yield ptr
            ptr = ptr.child

    def parent_iterator(self):
        ptr = self.parent
        while ptr:
            yield ptr
            ptr = ptr.parent


def get_line_bbox(line: TextLine) -> Rectangle:
    mins = np.min(line.polygon, axis=0)
    maxs = np.max(line.polygon, axis=0)

    # (minx, miny, maxx, maxy)
    return Rectangle(mins[0], mins[1], maxs[0], maxs[1])


def get_line_width(line: TextLine) -> float:
    bbox = get_line_bbox(line)
    return bbox.xmax - bbox.xmin


def get_line_height(line: TextLine) -> float:
    bbox = get_line_bbox(line)
    return bbox.ymax - bbox.ymin


def get_line_center(line: TextLine) -> Tuple[float, float]:
    bbox = get_line_bbox(line)
    x = (bbox.xmin + ((bbox.xmax - bbox.xmin) / 2))
    y = (bbox.ymin + ((bbox.ymax - bbox.ymin) / 2))
    return x, y


def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return sqrt(dx*dx + dy*dy)


def create_dataset(lines: List[Line], page_size, args, bert, tokenizer, device):
    samples = []
    with open(args.mapping, "r") as f:
        mapping_lines = f.readlines()
    mapping_lines = [l for l in mapping_lines if l.strip()]

    page_height, page_width = page_size
    
    for line in lines:
        features = []

        right_context = ""
        if line.child:
            right_context = line.child.text_line.transcription.strip()
        bert_embedding = get_embedding(
            line.text_line.transcription.strip(), 
            bert=bert, 
            tokenizer=tokenizer,
            device=device,
            right_context=right_context,
        )

        neighbourhood = [line]
        if line.parent:
            neighbourhood.append(line.parent)
            if line.parent.parent:
                neighbourhood.append(line.parent.parent)

        if line.child:
            neighbourhood.append(line.child)
            if line.child.child:
                neighbourhood.append(line.child.child)

        line_bbox = get_line_bbox(line.text_line)

        # Line width relative to it's neighbourhood
        avg_width = sum([get_line_width(l.text_line) for l in neighbourhood]) / len(neighbourhood)
        features.append(get_line_width(line.text_line) / avg_width)

        # Line width relative to all lines on page
        avg_width_page = sum(get_line_width(l.text_line) for l in lines) / len(lines)
        features.append(get_line_width(line.text_line) / avg_width_page)

        # Line height relative to all lines on page
        avg_height_page = sum(get_line_height(l.text_line) for l in lines) / len(lines)
        features.append(get_line_height(line.text_line) / avg_height_page)

        # Absolute distance from parent
        line_center = get_line_center(line.text_line)
        distance_to_parent = -100.0
        if line.parent:
            parent_center = get_line_center(line.parent.text_line)
            distance_to_parent = dist(line_center, parent_center)
        features.append(distance_to_parent)

        # Absolute distance from child
        distance_to_child = -100.0
        if line.child:
            child_center = get_line_center(line.child.text_line)
            distance_to_child = dist(line_center, child_center)
        features.append(distance_to_child)

        # Relative distance from parent relative to my height
        features.append(distance_to_parent / get_line_height(line.text_line))

        # Relative distance from child relative to my height
        features.append(distance_to_child / get_line_height(line.text_line))

        # Number of lines above me
        features.append(sum([1 for _ in line.parent_iterator()]))

        # Number of lines below me
        features.append(sum([1 for _ in line.children_iterator()]))

        # Line width relative to page width
        features.append(get_line_width(line.text_line) / page_width)

        # Line height relative to page height
        features.append(get_line_height(line.text_line) / page_height)

        # Center X coordinate relative to page width
        features.append(line_center[0] / page_width)

        # Center Y coordinate relative to page width
        features.append(line_center[1] / page_height)

        # Distance from parent in Y axis
        y_distance_from_parent = -100.0
        if line.parent:
            parent_y = parent_center[1]
            y_distance_from_parent = abs(line_center[1] - parent_y)
        features.append(y_distance_from_parent)

        # Distance from child in Y axis
        y_distance_from_child = -100
        if line.child:
            child_y = child_center[1]
            y_distance_from_child = abs(line_center[1] - child_y)
        features.append(y_distance_from_child)

        # Distance from parent in X axis
        x_distance_from_parent = -100.0
        if line.parent:
            parent_x = parent_center[0]
            x_distance_from_parent = abs(line_center[0] - parent_x)
        features.append(x_distance_from_parent)

        # Distance from child in X axis
        x_distance_from_child = -100
        if line.child:
            child_x = child_center[0]
            x_distance_from_child = abs(line_center[0] - child_x)
        features.append(x_distance_from_child)

        # Bounding box area relative to page area
        bb_area = (abs(line_bbox.xmax - line_bbox.xmin)) * (abs(line_bbox.ymax - line_bbox.ymin))
        features.append(bb_area / (page_width * page_height))

        # Bounding box area relative to bounding box area of parent
        bb_to_parent = -100.0
        if line.parent:
            bb_parent = get_line_bbox(line.parent.text_line)
            bb_parent_area = (abs(bb_parent.xmax - bb_parent.xmin)) * (abs(bb_parent.ymax - bb_parent.ymin))
            bb_to_parent = bb_area / bb_parent_area
        features.append(bb_to_parent)

        # Bounding box area relative to bounding box area of child
        bb_to_child = -100.0
        if line.child:
            bb_child = get_line_bbox(line.child.text_line)
            bb_child_area = (abs(bb_child.xmax - bb_child.xmin)) * (abs(bb_child.ymax - bb_child.ymin))
            bb_to_child = bb_area / bb_child_area
        features.append(bb_to_child)

        # Line width / line height
        features.append(get_line_width(line.text_line) / get_line_height(line.text_line))

        # Bounding box coordinates relative to page size
        features.append(line_bbox.xmin / page_width)
        features.append(line_bbox.xmax / page_width)
        features.append(line_bbox.ymin / page_height)
        features.append(line_bbox.ymax / page_height)
        
        # Find label
        for ml in mapping_lines:
            mltext, _label = ml.split("\t")
            if mltext.strip() == line.text_line.transcription.strip():
                label = _label
                break

        # print([f"{val:.2f}" for val in features])

        features = torch.tensor(features, dtype=torch.float32)
        combined_features = torch.cat([bert_embedding, features]) 
        try:
            samples.append(Sample(combined_features, LineLabel(int(label))))
        except UnboundLocalError:
            continue

    return samples


def main(args):
    # os.makedirs(args.save, exist_ok=True)
    image_filenames = [filename for filename in os.listdir(args.images) if filename.endswith(".jpg")]

    #################################################################################
    # EMBEDDINGS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model:
        checkpoint = torch.load(args.model)
        tokenizer = build_tokenizer(
            seq_len=checkpoint["seq_len"],
            fixed_sep=checkpoint["fixed_sep"],
            masking_prob=0.0,
        )

        bert = build_model(
            czert=checkpoint["czert"],
            vocab_size=len(tokenizer),
            device=device,
            seq_len=checkpoint["seq_len"],
            out_features=checkpoint["features"],
            mlm_level=0,
            sep=checkpoint["sep"],
        )
        bert.bert.load_state_dict(checkpoint["bert_state_dict"])
        bert.nsp_head.load_state_dict(checkpoint["nsp_head_state_dict"])
    else:
        tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
        bert = BertModel.from_pretrained(CZERT_PATH)

    bert = bert.to(device)
    #################################################################################

    samples = []
    for image_filename in image_filenames:
        img_path = os.path.join(args.images, image_filename)
        xml_path = os.path.join(args.xml, image_filename.replace(".jpg", ".xml"))
        img = cv2.imread(img_path)

        pagexml = PageLayout(file=xml_path)

        lines = [Line(line, region) for region in pagexml.regions for line in region.lines]
        for line in lines:
            line.set_parent(lines)
            line.set_child(lines)

        samples.extend(create_dataset(lines, pagexml.page_size, args, bert, tokenizer, device))
        
    with open("data-combined-lm72.pkl", "wb") as f:
        pickle.dump(samples, f)

        # for line in lines:
        #     if line.parent: # PARENT RED
        #         start = (int(line.center[0]) + 20, int(line.center[1]))
        #         end = (int(line.parent.center[0]) + 20, int(line.parent.center[1]))
        #         cv2.line(img, start, end, (0, 0, 255), 7)

        #     if line.child: # CHILD GREEN
        #         start = (int(line.center[0]), int(line.center[1]))
        #         end = (int(line.child.center[0]), int(line.child.center[1]))
        #         cv2.line(img, start, end, (0, 255, 0), 7)

        # pth = os.path.join(args.save, image_filename.replace(".jpg", "-geo.jpg"))
        # cv2.imwrite(pth, img)


if __name__ == "__main__":
    args = parse_arguments()
    safe_gpu.claim_gpus()
    main(args)
