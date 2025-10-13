import argparse
import sys
import os
import logging
import json
from tqdm import tqdm

import cv2

from textbite.geometry import PageGeometry, bbox_center, AABB, bbox_intersection_over_area, bbox_area
from textbite.visualization.utils import COLORS

COLORS_ = COLORS.copy()
COLORS_.remove((0, 0, 255))
# print(COLORS_)


METADATA_LABEL = 3


logger = logging.getLogger(__name__)


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--coco", required=True, type=str, help="Path to a file with TextBite COCO annotations.")
    parser.add_argument("--img", type=str, help="Path to a folder with images.")
    parser.add_argument("--save-paragraph-yolo", action="store_true", help="Save paragraphs in YOLO format.")
    parser.add_argument("--output", required=True, type=str, help="Path to the output folder.")

    args = parser.parse_args()
    return args


def visualize_mapping(annotations, relations, mapped_annotations, unmapped_paragraphs, img_path):
    img = cv2.imread(img_path)
    img_original = img.copy()

    # Draw original annotations
    for annotation in annotations:
        xmin, ymin, w, h = annotation["bbox"]
        x_max = xmin + w
        y_max = ymin + h
        cv2.rectangle(img_original, (int(xmin), int(ymin)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    for relation in relations:
        from_annotation_id = relation["from_annotation_id"]
        to_annotation_id = relation["to_annotation_id"]

        from_annotation = next((a for a in annotations if a["id"] == from_annotation_id), None)
        to_annotation = next((a for a in annotations if a["id"] == to_annotation_id), None)

        assert from_annotation is not None
        assert to_annotation is not None

        from_xmin, from_ymin, from_w, from_h = from_annotation["bbox"]
        from_x_max = from_xmin + from_w
        from_y_max = from_ymin + from_h

        to_xmin, to_ymin, to_w, to_h = to_annotation["bbox"]
        to_x_max = to_xmin + to_w
        to_y_max = to_ymin + to_h

        # Draw arrow from the center of the from_annotation to the center of the to_annotation
        from_center = bbox_center(AABB(from_xmin, from_ymin, from_x_max, from_y_max))
        to_center = bbox_center(AABB(to_xmin, to_ymin, to_x_max, to_y_max))
        cv2.arrowedLine(img_original, (int(from_center.x), int(from_center.y)), (int(to_center.x), int(to_center.y)), (255, 0, 0), 2, tipLength=0.1)

    # Draw new mapping
    for _, bbox in unmapped_paragraphs:
        cv2.rectangle(img, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 0, 255), 2)

    for i, (idx, mapping) in enumerate(mapped_annotations.items()):
        COLOR = COLORS_[i % len(COLORS_)]

        annotation = next((a for a in annotations if a["id"] == idx), None)
        assert annotation is not None

        xmin, ymin, w, h = annotation["bbox"]
        x_max = xmin + w
        y_max = ymin + h
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(x_max), int(y_max)), COLOR, 10)

        for _, bbox in mapping:
            cv2.rectangle(img, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), COLOR, 2)

    img = cv2.hconcat([img_original, img])

    return img


def create_paragraphs_from_geometry(
    geometry: PageGeometry,
) -> list[AABB]:
    # Create paragraphs
    children_lines = [line.child for line in geometry.lines if line.child is not None]
    first_lines = list(set(geometry.lines) - set(children_lines))
    paragraph_lines = [[line] + [i for i in line.children_iterator()] for line in first_lines]
    paragraphs = []

    for paragraph in paragraph_lines:
        x_min = min(line.bbox.xmin for line in paragraph)
        y_min = min(line.bbox.ymin for line in paragraph)
        x_max = max(line.bbox.xmax for line in paragraph)
        y_max = max(line.bbox.ymax for line in paragraph)

        transcription = " ".join(line.text_line.transcription.strip() for line in paragraph if line.text_line.transcription)

        # Find the best matching annotation bbox for the current paragraph
        bbox = AABB(x_min, y_min, x_max, y_max)
        paragraphs.append((bbox, transcription))

    return paragraphs


def remove_inter_title_annotations(coco_data, file_entry):
    # Filter first annotations and relations for the current image
    relations = [relation for relation in coco_data["relations"] if relation["image_id"] == file_entry["id"]]
    annotations = [annotation for annotation in coco_data["annotations"] if annotation["image_id"] == file_entry["id"]]

    annotation_ids_to_remove = []
    for annotation in annotations:
        other_annotations = [a for a in annotations if a["id"] != annotation["id"]]
        xmin, ymin, w, h = annotation["bbox"]
        x_max = xmin + w
        y_max = ymin + h
        annotation_bbox = AABB(xmin, ymin, x_max, y_max)

        for other_annotation in other_annotations:
            other_xmin, other_ymin, other_w, other_h = other_annotation["bbox"]
            other_x_max = other_xmin + other_w
            other_y_max = other_ymin + other_h
            other_annotation_bbox = AABB(other_xmin, other_ymin, other_x_max, other_y_max)

            if bbox_area(annotation_bbox) > bbox_area(other_annotation_bbox):
                continue

            intersection = bbox_intersection_over_area(annotation_bbox, other_annotation_bbox)
            if intersection > 0.5:
                annotation_ids_to_remove.append(annotation["id"])
                other_annotation["includes_title"] = True
                break

    annotations = [annotation for annotation in annotations if annotation["id"] not in annotation_ids_to_remove]
    relations = [relation for relation in relations if relation["from_annotation_id"] not in annotation_ids_to_remove and relation["to_annotation_id"] not in annotation_ids_to_remove]

    return annotations, relations


def build_annotation_chain(annotations, relations):
    # Remove annotations that have relation pointing to them
    annotation_chain = [[annotation] for annotation in annotations if not any(relation["to_annotation_id"] == annotation["id"] for relation in relations)]

    for idx in range(len(annotation_chain)):
        annotation = annotation_chain[idx][0]
        annotation_id = annotation["id"]

        visited_ids = set()
        visited_ids.add(annotation_id)
        found_relation = next((relation for relation in relations if relation["from_annotation_id"] == annotation_id), None)
        while found_relation is not None:
            tgt_id = found_relation["to_annotation_id"]
            if tgt_id in visited_ids:
                logger.error(f"Found a cycle in the relations for annotation {annotation_id}. Quitting.")
                exit(1)
            visited_ids.add(tgt_id)

            tgt_annotation = next((a for a in annotations if a["id"] == tgt_id), None)
            if tgt_annotation is None:
                logger.error(f"Annotation {tgt_id} not found for relation, something went seriously wrong. Quitting.")
                exit(1)

            annotation_chain[idx].append(tgt_annotation)
            found_relation = next((relation for relation in relations if relation["from_annotation_id"] == tgt_annotation["id"]), None)

    return annotation_chain


def map_paragraphs_to_annotations(annotation_chain, paragraphs):
    paragraph_chains = [[] for _ in annotation_chain]
    paragraph_labels = [[] for _ in annotation_chain]

    annotations_inner_title = [False] * len(annotation_chain)
    for chain_idx, chain in enumerate(annotation_chain):
        for annotation in chain:
            xmin, ymin, w, h = annotation["bbox"]
            x_max = xmin + w
            y_max = ymin + h
            annotation_bbox = AABB(xmin, ymin, x_max, y_max)

            annotation_label = annotation["category_id"]

            if "includes_title" in annotation:
                if not annotation["includes_title"]:
                    logger.error(f"Annotation {annotation['id']} has 'includes_title' set to False, but it is not a title. Quitting.")
                    exit(1)
                annotations_inner_title[chain_idx] = True

            new_paragraphs = []
            to_delete_paraghaph_indices = []
            for i, (paragraph_bbox, paragraph_transcription) in enumerate(paragraphs):
                intersection = bbox_intersection_over_area(paragraph_bbox, annotation_bbox) if bbox_area(paragraph_bbox) < bbox_area(annotation_bbox) else bbox_intersection_over_area(annotation_bbox, paragraph_bbox)
                if intersection > 0.5:
                    new_paragraphs.append((paragraph_bbox, annotation_label, paragraph_transcription))
                    to_delete_paraghaph_indices.append(i)

            new_paragraphs = sorted(new_paragraphs, key=lambda p: p[0].ymin)
            p = [(para, transcription) for para, _, transcription in new_paragraphs]
            labels = [label for _, label, _ in new_paragraphs]
            paragraph_chains[chain_idx].extend(p)
            paragraph_labels[chain_idx].extend(labels)
            paragraphs = [paragraphs[i] for i in range(len(paragraphs)) if i not in to_delete_paraghaph_indices]

    return paragraph_chains, paragraph_labels, annotations_inner_title, paragraphs


def create_annotations(paragraphs, paragraph_chains, paragraph_labels, annotations_inner_title, file_entry, result, running_annotation_id, running_relation_id):
    for paragrah_chain, labels, has_title in zip(paragraph_chains, paragraph_labels, annotations_inner_title):
        if len(paragrah_chain) == 0:
            continue
        first_item = paragrah_chain[0]
        # print(first_item[1])
        result["annotations"].append({
            "id": running_annotation_id[0],
            "image_id": file_entry["id"],
            "category_id": labels[0],
            "bbox": [first_item[0].xmin, first_item[0].ymin, first_item[0].xmax - first_item[0].xmin, first_item[0].ymax - first_item[0].ymin],
            "area": (first_item[0].xmax - first_item[0].xmin) * (first_item[0].ymax - first_item[0].ymin),
            "iscrowd": 0,
            "segmentation": [],
            "includes_title": has_title if labels[0] != 2 else False,
            "transcription": first_item[1]
        })
        running_annotation_id[0] += 1

        for to_paragraph, label in zip(paragrah_chain[1:], labels[1:]):
            result["relations"].append({
                "id": running_relation_id[0],
                "image_id": file_entry["id"],
                "from_annotation_id": running_annotation_id[0] - 1,
                "to_annotation_id": running_annotation_id[0],
            })
            running_relation_id[0] += 1

            result["annotations"].append({
                "id": running_annotation_id[0],
                "image_id": file_entry["id"],
                "category_id": label,
                "bbox": [to_paragraph[0].xmin, to_paragraph[0].ymin, to_paragraph[0].xmax - to_paragraph[0].xmin, to_paragraph[0].ymax - to_paragraph[0].ymin],
                "area": (to_paragraph[0].xmax - to_paragraph[0].xmin) * (to_paragraph[0].ymax - to_paragraph[0].ymin),
                "iscrowd": 0,
                "segmentation": [],
                "includes_title": False,
                "transcription": to_paragraph[1],
            })
            running_annotation_id[0] += 1

    for leftover_paragraph in paragraphs:
        result["annotations"].append({
            "id": running_annotation_id[0],
            "image_id": file_entry["id"],
            "category_id": METADATA_LABEL,
            "bbox": [leftover_paragraph[0].xmin, leftover_paragraph[0].ymin, leftover_paragraph[0].xmax - leftover_paragraph[0].xmin, leftover_paragraph[0].ymax - leftover_paragraph[0].ymin],
            "area": (leftover_paragraph[0].xmax - leftover_paragraph[0].xmin) * (leftover_paragraph[0].ymax - leftover_paragraph[0].ymin),
            "iscrowd": 0,
            "segmentation": [],
            "includes_title": False,
            "transcription": leftover_paragraph[1],
        })
        running_annotation_id[0] += 1


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    os.makedirs(args.output, exist_ok=True)

    with open(args.coco, 'r') as coco_file:
        coco_data = json.load(coco_file)
    coco_data["images"] = [image for image in coco_data["images"]]# if image["file_name"] == "4ca7a7ce-042d-49b0-9969-573a5ab4d40f.jpg"]

    # Create top level structure for the result
    result = {"info": coco_data["info"], "images": coco_data["images"], "categories": coco_data["categories"], "annotations": [], "relations": []}
    result["categories"].append({"id": 3, "name": "unannotated", "supercategory": "object"})
    
    running_annotation_id = [0]
    running_relation_id = [0]

    for file_entry in tqdm(coco_data["images"]):
    # for file_entry in coco_data["images"]:
        filename = file_entry["file_name"]

        # # DEBUG
        # if filename != "4ca7a7ce-042d-49b0-9969-573a5ab4d40f.jpg":
        #     continue

        xml_path = os.path.join(args.xml, filename.replace('.jpg', '.xml'))

        geometry = PageGeometry(path=xml_path)        
        geometry.split_geometry_into_paragraphs()

        paragraphs = create_paragraphs_from_geometry(geometry) # paragrahs: list[tuple[AABB, str]]
        annotations, relations = remove_inter_title_annotations(coco_data, file_entry)
        annotation_chain = build_annotation_chain(annotations, relations)
        paragraph_chains, paragraph_labels, annotations_inner_title, leftover_paragraphs = map_paragraphs_to_annotations(annotation_chain, paragraphs)

        create_annotations(leftover_paragraphs, paragraph_chains, paragraph_labels, annotations_inner_title, file_entry, result, running_annotation_id, running_relation_id)

    with open(os.path.join(args.output, "coco.json"), 'w') as output_file:
        json.dump(result, output_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
