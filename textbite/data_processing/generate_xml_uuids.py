"""Utility script to modify PAGE-XML to ensure UUIDs as line ids

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


import sys
import os
import uuid
import argparse

from pero_ocr.core.layout import PageLayout


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--xml", required=True, type=str, help="Path to a folder containing XML files from PERO-OCR.")
    parser.add_argument("--save", default=".", type=str, help="Where to store results.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    os.makedirs(args.save, exist_ok=True)

    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xml) if xml_filename.endswith(".xml")]

    for xml_filename in xml_filenames:
        xml_path = os.path.join(args.xml, xml_filename)

        pagexml = PageLayout(file=xml_path)

        for line in pagexml.lines_iterator():
            line.id = str(uuid.uuid4())

        save_path = os.path.join(args.save, xml_filename)
        pagexml.to_pagexml(save_path)


if __name__ == "__main__":
    main()
