import argparse
import os
import logging
from itertools import pairwise
import pickle
import random

from transformers import BertTokenizerFast
import torch

from numba.core.errors import NumbaDeprecationWarning
import warnings
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

from pero_ocr.document_ocr.layout import PageLayout

from textbite.data_processing.label_studio import LabelStudioExport, AnnotatedDocument
from textbite.utils import CZERT_PATH


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xmls", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--tokenizer", default=CZERT_PATH, type=str, help="Path to a tokenizer.")
    parser.add_argument("--export", required=True, type=str, help="Path to the Label-Studio export.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output pickles.")
    parser.add_argument("--filename", required=True, type=str, help="Output file name.")

    return parser.parse_args()


def process_document(annotated_doc: AnnotatedDocument, tokenizer: BertTokenizerFast) -> list:
    linear_lines = [line for region in annotated_doc.regions for line in region.lines]

    positive_samples = []
    negative_samples = []
    for region in annotated_doc.regions:
        for top_line, bot_line in pairwise(region.lines):
            top_text = top_line.transcription.strip()
            bot_text = bot_line.transcription.strip()

            tokenized_text = tokenizer(
                top_text,
                bot_text,
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )
            tokenized_text["label"] = torch.tensor([1])
            positive_samples.append(tokenized_text)

            # Create negative example for the top_line
            other_lines = [line for line in linear_lines if line is not bot_line]
            bot_line_negative = random.choice(other_lines)
            bot_line_negative_text = bot_line_negative.transcription.strip()
            tokenized_text = tokenizer(
                top_text,
                bot_line_negative_text,
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )
            tokenized_text["label"] = torch.tensor([0])
            negative_samples.append(tokenized_text)

    assert len(positive_samples) == len(negative_samples)
    return positive_samples + negative_samples


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    export = LabelStudioExport(path=args.export)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    data = []
    file_idx = 1

    for idx, annotated_doc in enumerate(export.documents):
        xml_filename = annotated_doc.filename.replace(".jpg", ".xml")
        xml_path = os.path.join(args.xmls, xml_filename)
        
        logging.info(f"{idx}/{len(export.documents)} | Processing {xml_path}.")

        try:
            pagexml = PageLayout(file=xml_path)
        except OSError:
            logging.debug(f"XML {xml_path} not in this folder, skipping.")
            continue

        annotated_doc.map_to_pagexml(pagexml=pagexml)

        document_results = process_document(annotated_doc=annotated_doc, tokenizer=tokenizer)
        data.extend(document_results)

        if (idx + 1) % 500 == 0:
            save_path = os.path.join(args.save, args.filename.replace(".pkl", f"-{file_idx}.pkl"))
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
            data = []
            file_idx += 1


if __name__ == "__main__":
    main()
