import argparse
import os
import sys

from pero_ocr.document_ocr.layout import PageLayout
from filter_xml import filter_pagelayout


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a folder containing xml data.")

    args = parser.parse_args()
    return args


def main(args):
    for filename in os.listdir(args.data):
        if not filename.endswith(".xml"):
            continue

        filename_txt = f"{filename[:-3]}txt"
        string = ""

        path = os.path.join(args.data, filename)
        pagexml = PageLayout(file=path)
        pagexml = filter_pagelayout(pagexml)

        for region in pagexml.regions:
            for line in region.lines:
                string += f"{line.transcription}\n"
            string += "\n"

        string = string[:-1]

        with open(os.path.join(args.data, filename_txt), "w") as f:
            f.write(string)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
