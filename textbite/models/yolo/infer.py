#!/usr/bin/env python3

import argparse
import json
import logging
import os.path
from typing import List

from pero_ocr.document_ocr.layout import PageLayout, TextLine
from ultralytics import YOLO

from textbite.data_processing.label_studio import best_intersecting_bbox
from textbite.geometry import AABB, polygon_to_bbox


class YoloBiter:
    def __init__(self, model):
        self.model = model

    def produce_bites(self, img_fn, xml_fn):
        results = self.model.predict(source=img_fn)
        assert len(results) == 1
        result = results[0]

        cls_indices = {v: k for k, v in result.names.items()}
        bite_index = cls_indices['bite']

        bites = [b for b in result.boxes if b.cls == bite_index]
        for b in bites:
            b.bbox = AABB(*b.xyxy[0].numpy().tolist())

        layout = PageLayout()
        with open(xml_fn) as f:
            layout.from_pagexml(f)

        bites_lines = [[] for _ in bites]
        for line in layout.lines_iterator():
            line_bbox = polygon_to_bbox(line.polygon)
            best_bite_idx = best_intersecting_bbox(line_bbox, [b.bbox for b in bites])
            if best_bite_idx is None:
                continue
            bites_lines[best_bite_idx].append(line.id)

        return bites_lines


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--data", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--images", type=str, help="Path to a folder with images data. Used for extra visualizations.")
    parser.add_argument("--diagnostics", type=str, help="Path to a folder where to place diagnostic images. IGNORED")
    parser.add_argument("--model", required=True, type=str, help="Path to the .pt file with weigts of YOLO model.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")

    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=args.logging_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    biter = YoloBiter(YOLO(args.model))

    os.makedirs(args.save, exist_ok=True)
    xml_filenames = [xml_filename for xml_filename in os.listdir(args.data) if xml_filename.endswith(".xml")]

    for filename in xml_filenames:
        path_xml = os.path.join(args.data, filename)
        path_img = os.path.join(args.images, filename.replace(".xml", ".jpg"))

        logging.info(f"Processing: {path_xml}")
        result = biter.produce_bites(path_img, path_xml)

        out_path = os.path.join(args.save, filename.replace(".xml", ".json"))
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)

        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()
