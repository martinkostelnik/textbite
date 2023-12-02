import sys
import argparse
import os
import logging

import numpy as np

from pero_ocr.document_ocr.layout import PageLayout

from textbite.utils import LineLabel
from textbite.data_processing.label_studio import LabelStudioExport, AnnotatedDocument, RegionType


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True, type=str, help="Path to an exported JSON file from label-studio.")
    parser.add_argument('--xml', required=True, type=str, help="Path to a folder containing XML files from PERO-OCR.")
    parser.add_argument("--ignore-relations", action="store_true", help="If set, regions which further continue elsewhere will still be terminal")

    args = parser.parse_args()
    return args


def print_labels(annotated_document: AnnotatedDocument, ignore_relations: bool):
    for region in annotated_document.regions:
        match region.label:
            case RegionType.TITLE:
                for line in region.lines:
                    print(f"{line.transcription.strip()}\t{LineLabel.TITLE.value}")

            case RegionType.TEXT:
                lowest_line = min(region.lines, key=lambda x: np.max(x.polygon, axis=0)[1])

                other_lines = [line for line in region.lines if line is not lowest_line]
                for line in other_lines:
                    print(f"{line.transcription.strip()}\t{LineLabel.NONE.value}")

                label = LineLabel.TERMINATING
                if not ignore_relations and region.id in annotated_document.relations.keys():
                    label = LineLabel.NONE
                print(f"{lowest_line.transcription.strip()}\t{label.value}")

            case _:
                logging.warning(f"Unknown region: {region.label}")

        print()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

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

        annotated_document.map_to_pagexml(pagexml)
        print_labels(annotated_document, args.ignore_relations)


if __name__ == "__main__":
    main()
