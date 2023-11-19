""" Sorts xml files into reading order and saves them as both .txt and .xml files.

    Pipeline: raw_xml -> filtered_xml -> sort -> cross reference with original xlm -> save txt, xml
"""


import argparse
import sys
import os
import time
import shutil

from pero_ocr.document_ocr.layout import PageLayout

from reading_order.document import page_xml
from reading_order.language_model.analyzer import LmAnalyzer
from reading_order.language_model.carrier import cs_model, cs_vocab
from reading_order.spatial.analyzer import ColumnarLmAnalyzer
from reading_order.reading_order.reading_order import ReadingOrder

from filter_xml import filter_pagelayout


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", "-d", required=True, type=str, help="Path to a folder containing data.")
    parser.add_argument('--tokens', '-t', type=int, help='Number of tokens, using for L and C analyse, default=3', default=7)
    parser.add_argument("--out", required=True, type=str, help="Path to output folder.")

    args = parser.parse_args()
    return args


def add_newlines(pagexml: PageLayout):
    for region in pagexml.regions:
        for line in region.lines:
            line.transcription += "\n"


def sort(path: str, analyzer, lm_analyzer) -> ReadingOrder:
    doc = page_xml.parse(path)
    ro = analyzer.analyze(doc, lm_analyzer)
    return ro


def main(args):
    # Create TMP folder for xml filtering
    TMP_FOLDER_NAME = f"tmp-{int(time.time())}"
    os.makedirs(TMP_FOLDER_NAME)

    # Load model for sorting
    model, vocab = cs_model(), cs_vocab()
    lm_analyzer = LmAnalyzer(model, vocab)
    # lm_analyzer.use_hard_limit(args.tokens)
    lm_analyzer.use_score_hard_limit(args.tokens)
    analyzer = ColumnarLmAnalyzer()

    for filename in os.listdir(args.data):
        if not filename.endswith(".xml"):
            continue

        # Filter XML and save to tmp folder
        path = os.path.join(args.data, filename)
        pagexml = PageLayout(file=path)
        add_newlines(pagexml)
        pagexml = filter_pagelayout(pagexml)
        filtered_path = os.path.join(TMP_FOLDER_NAME, filename)
        pagexml.to_pagexml(file_name=filtered_path)

        # Read XML from tmp folder and sort it
        try:
            reading_order = sort(filtered_path, analyzer, lm_analyzer)
        except IndexError:
            continue

        # Save sorted XML as txt
        out_path = os.path.join(args.out, f"{filename[:-3]}txt")
        with open(out_path, "w") as f:
            for idx, i in enumerate(reading_order.get_all_items()):
                if idx == len(reading_order.get_all_items()) - 1:
                    print(f"{i.text_region.get_text_raw()}", file=f, end="")
                else:
                    print(f"{i.text_region.get_text_raw()}", file=f)

    # Remove tmp folder
    shutil.rmtree(TMP_FOLDER_NAME, ignore_errors=True)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
