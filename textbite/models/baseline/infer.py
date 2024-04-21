import argparse
import os
import logging
from typing import List
from enum import Enum
from itertools import pairwise

from numba.core.errors import NumbaDeprecationWarning
import warnings
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

from safe_gpu import safe_gpu
import torch

from textbite.language_model import create_language_model
from textbite.geometry import PageGeometry, LineGeometry, enclosing_bbox, bbox_dist_y
from textbite.bite import Bite, save_bites


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xmls", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--model", type=str, help="Path to the .pt file with weights of language model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")
    parser.add_argument("--method", choices=["lm", "dist", "base"], default="base", help="One of [method, dist, base].")

    return parser.parse_args()


class SplitMethod(Enum):
    BASE = "base"
    LM = "lm"
    DIST = "dist"


def create_bite(lines: List[LineGeometry]) -> Bite:
    lines_ids = [line.text_line.id for line in lines]
    lines_bboxes = [line.bbox for line in lines]
    bbox = enclosing_bbox(lines_bboxes)

    return Bite("text", bbox, lines_ids)


def lm_forward(top_line: LineGeometry, bot_line: LineGeometry, threshold: float, lm, tokenizer, device) -> bool:
    top_text = top_line.text_line.transcription.strip()
    bot_text = bot_line.text_line.transcription.strip()

    tokenized = tokenizer(top_text, bot_text)

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    token_type_ids = tokenized["token_type_ids"].to(device)

    with torch.no_grad():
        bert_output = lm(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    probability = bert_output.nsp_output.cpu().item()
    return probability < threshold


def dist_forward(top_line: LineGeometry, bot_line: LineGeometry, threshold: float) -> bool:
    return bbox_dist_y(top_line.bbox, bot_line.bbox) > threshold


def get_base_bites(geometry: PageGeometry) -> List[Bite]:
    bites = []
    
    for head_line in geometry.line_heads:
        column = [head_line] + [line for line in head_line.children_iterator()]
        bite = create_bite(column)
        bites.append(bite)
    
    return bites


def get_split_bites(
        geometry: PageGeometry,
        threshold: float,
        break_predicate,
        **kwargs,
    ) -> List[Bite]:
    bites = []
    for head_line in geometry.line_heads:
        column = [head_line] + [line for line in head_line.children_iterator()]

        if len(column) == 1:
            bites.append(create_bite(column))

        lines: List[LineGeometry] = []

        for top_line, bot_line in pairwise(column):
            top_text = top_line.text_line.transcription.strip()
            bot_text = bot_line.text_line.transcription.strip()

            if top_line not in lines:
                lines.append(top_line)

            if top_text == "":
                if top_line not in lines:
                    lines.append(top_line)
                continue

            if bot_text == "":
                if bot_line not in lines:
                    lines.append(bot_line)
                continue

            if break_predicate(top_line, bot_line, threshold, **kwargs): # Break the bite
                bite = create_bite(lines)
                bites.append(bite)
                lines = [bot_line]
            else:
                if bot_line not in lines:
                    lines.append(bot_line)

        if lines:
            bite = create_bite(lines)
            bites.append(bite)

    assert len(geometry.lines) == sum([len(bite.lines) for bite in bites])
    return bites


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    method = SplitMethod(args.method)

    if method == SplitMethod.LM:
        safe_gpu.claim_gpus()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Inference running on {device}")

        logging.info("Loading language model ...")
        tokenizer, bert = create_language_model(device, args.model)
        bert.eval()
        bert = bert.to(device)
        logging.info("Language model loaded.")

        threshold = args.threshold

    logging.info("Creating save directory ...")
    os.makedirs(args.save, exist_ok=True)
    logging.info("Save directory created.")

    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xmls) if xml_filename.endswith(".xml")]

    for i, xml_filename in enumerate(xml_filenames):
        json_filename = xml_filename.replace(".xml", ".json")
        path_xml = os.path.join(args.xmls, xml_filename)
        save_path = os.path.join(args.save, json_filename)

        logging.info(f"({i}/{len(xml_filenames)}) | Processing: {path_xml}")

        geometry = PageGeometry(path=path_xml)
        geometry.split_geometry()

        match method:
            case SplitMethod.BASE:
                bites = get_base_bites(geometry)

            case SplitMethod.DIST:
                threshold = geometry.avg_line_distance_y * args.threshold
                bites = get_split_bites(geometry, threshold, dist_forward)

            case SplitMethod.LM:
                bites = get_split_bites(geometry, threshold, lm_forward, lm=bert, tokenizer=tokenizer, device=device)

            case _:
                raise ValueError("Invalid split method detected.")

        save_bites(bites, save_path)


if __name__ == "__main__":
    main()
