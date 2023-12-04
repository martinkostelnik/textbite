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

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--json", required=True, type=str, help="Path to an exported JSON file from label-studio.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder containing XML files from PERO-OCR.")
    parser.add_argument("--ignore-relations", action="store_true", help="If set, regions which further continue elsewhere will still be terminal")
    parser.add_argument("--print", action="store_true", help="Prints transcriptions and labels to stdout.")
    parser.add_argument("--save", default=".", type=str, help="Where to store results.")

    args = parser.parse_args()
    return args


def create_labels(annotated_document: AnnotatedDocument, ignore_relations: bool, p: bool):
    result = ""
    for region in annotated_document.regions:
        match region.label:
            case RegionType.TITLE:
                for line in region.lines:
                    if p:
                        print(f"{line.transcription.strip()}\t{LineLabel.TITLE.value}")
                    result += f"{line.id}\t{LineLabel.TITLE.value}\n"

            case RegionType.TEXT:
                lowest_line = min(region.lines, key=lambda x: np.max(x.polygon, axis=0)[1])

                other_lines = [line for line in region.lines if line is not lowest_line]
                for line in other_lines:
                    if p:
                        print(f"{line.transcription.strip()}\t{LineLabel.NONE.value}")
                    result += f"{line.id}\t{LineLabel.NONE.value}\n"

                label = LineLabel.TERMINATING
                if not ignore_relations and region.id in annotated_document.relations.keys():
                    label = LineLabel.NONE
                    
                if p:
                    print(f"{lowest_line.transcription.strip()}\t{label.value}")
                result += f"{lowest_line.id}\t{label.value}\n"

            case _:
                logging.warning(f"Unknown region: {region.label}")

        if p:
            print()

    return result


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    annotated_data = LabelStudioExport(args.json)

    result_str = ""
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
        result_str += create_labels(annotated_document, args.ignore_relations, args.print)

    save_path = os.path.join(args.save, "ids-to-labels.txt")
    with open(save_path, "w") as f:
        print(result_str, file=f, end="")


if __name__ == "__main__":
    main()
