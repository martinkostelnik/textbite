"""Transforms exported data file from Label-Studio to the YOLO format

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


import os
import sys
import argparse
import logging

from pero_ocr.core.layout import PageLayout

from textbite.data_processing.label_studio import LabelStudioExport


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--json", required=True, type=str, help="Path to an exported JSON file from label-studio.")
    parser.add_argument('--xml', required=True, type=str, help="Path to a folder containing XML files from PERO-OCR.")
    parser.add_argument("--save", required=True, type=str, help="Path to a folder where results will be saved.")
    parser.add_argument("--raw", action="store_true", help="Raw regions as they come from label studio.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    os.makedirs(args.save, exist_ok=True)

    annotated_data = LabelStudioExport(args.json)
    for annotated_document in annotated_data.documents:
        filename_img = annotated_document.filename
        filename_xml = filename_img.replace(".jpg", ".xml")

        path_xml = os.path.join(args.xml, filename_xml)
        try:
            pagexml = PageLayout(file=path_xml)
        except OSError:
            logging.warning(f"XML file {path_xml} not found. SKIPPING")
            continue

        if not args.raw:
            annotated_document.map_to_pagexml(pagexml)
        
        result_str = annotated_document.get_yolo_str()

        save_path = os.path.join(args.save, filename_img.replace(".jpg", ".txt"))
        with open(save_path, "w") as f:
            print(result_str, file=f, end="")


if __name__ == "__main__":
    main()
