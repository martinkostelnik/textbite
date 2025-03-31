"""Transforms exported data file from Label-Studio to the output JSON format

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


import sys
import argparse
import os
import logging

from pero_ocr.core.layout import PageLayout

from textbite.data_processing.label_studio import LabelStudioExport
from textbite.bite import save_bites


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--json", required=True, type=str, help="Path to an exported JSON file from label-studio.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder containing XML files from PERO-OCR.")
    parser.add_argument("--ignore-relations", action="store_true", help="If set, regions which further continue elsewhere will still be terminal")
    parser.add_argument("--save", default=".", type=str, help="Where to store results.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    annotated_data = LabelStudioExport(args.json)

    os.makedirs(args.save, exist_ok=True)

    result_str = ""
    for annotated_document in annotated_data.documents:
        filename_img = annotated_document.filename
        filename_xml = filename_img[:-len(".jpg")] + ".xml"
        filename_json = filename_img[:-len(".jpg")] + ".json"

        path_xml = os.path.join(args.xml, filename_xml)
        try:
            pagexml = PageLayout(file=path_xml)
        except OSError:
            logging.warning(f"XML file {path_xml} not found. SKIPPING")
            continue

        annotated_document.map_to_pagexml(pagexml)
        if not args.ignore_relations:
            annotated_document.merge_regions()

        bites = annotated_document.to_bites(["image", "image-desc", "page-number", "note"])

        save_path = os.path.join(args.save, filename_json)
        save_bites(bites, save_path)

    # save_path = os.path.join(args.save, "ids-to-bite-ids.txt")
    # with open(save_path, "w") as f:
    #     print(result_str, file=f, end="")


if __name__ == "__main__":
    main()
