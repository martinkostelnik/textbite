"""Utility for removing empty and corrupted lines from PAGE-XML

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


import argparse
import os
import sys
import copy

from pero_ocr.core.layout import PageLayout


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a folder containing xml data.")
    parser.add_argument("--out", required=True, type=str, help="Path to a folder to save output.")

    args = parser.parse_args()
    return args


def filter_lines(pagexml: PageLayout) -> PageLayout:
    _pagexml = copy.copy(pagexml)

    for region in _pagexml.regions:
        region.lines = [line for line in region.lines if line.transcription.strip() or line.transcription.strip() != ""]

    return _pagexml


def filter_regions(pagexml: PageLayout) -> PageLayout:
    _pagexml = copy.copy(pagexml)

    _pagexml.regions = [region for region in pagexml.regions if \
                        len(region.lines) > 0 \
                        and not (len(region.lines) == 1 and len(region.lines[0].transcription.strip()) < 5) \
                        and not all(line.transcription.strip().isdigit() for line in region.lines)]

    return _pagexml


def filter_pagelayout(pagexml: PageLayout) -> PageLayout:
    _pagexml = copy.copy(pagexml)

    _pagexml = filter_lines(_pagexml)
    _pagexml = filter_regions(_pagexml)

    return _pagexml


def main(args):
    for filename in os.listdir(args.data):
        if not filename.endswith(".xml"):
            continue

        path = os.path.join(args.data, filename)
        pagexml = PageLayout(file=path)
        pagexml = filter_pagelayout(pagexml)

        out_path = os.path.join(args.out, filename)
        pagexml.to_pagexml(file_name=out_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
