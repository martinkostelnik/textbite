"""Script for creating training graphs.

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


import argparse
import logging
import os
import pickle

from ultralytics import YOLO
import torch
from transformers import BertModel, BertTokenizerFast
from safe_gpu import safe_gpu

from pero_ocr.document_ocr.layout import PageLayout

from textbite.data_processing.label_studio import LabelStudioExport
from textbite.models.yolo.infer import YoloBiter
from textbite.models.joiner.graph import JoinerGraphProvider
from textbite.utils import CZERT_PATH


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--json", required=True, type=str, help="Path to label studio exported json.")
    parser.add_argument("--model", required=True, type=str, help="Path to the .pt file with weights of YOLO model.")
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--xmls", type=str, help="Path to a folder with xml data.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output pickle file.")
    parser.add_argument("--filename", default="graphs.pkl", type=str, help="Output filename.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    safe_gpu.claim_gpus()

    # Load CZERT and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Creating graphs on: {device}")

    logging.info("Loading tokenizer ...")
    tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
    logging.info("Tokenizer loaded.")
    logging.info("Loading CZERT ...")
    czert = BertModel.from_pretrained(CZERT_PATH)
    czert = czert.to(device)
    logging.info("CZERT loaded.")

    export = LabelStudioExport(args.json)
    yolo = YoloBiter(YOLO(args.model))
    graph_provider = JoinerGraphProvider(tokenizer, czert, device)

    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xmls) if xml_filename.endswith(".xml")]
    bad_files = 0
    graphs = []

    for i, xml_filename in enumerate(xml_filenames):
        img_filename = xml_filename.replace(".xml", ".jpg")
        path_xml = os.path.join(args.xmls, xml_filename)
        path_img = os.path.join(args.images, img_filename)
        
        logging.info(f"({i}/{len(xml_filenames)})Processing: {path_xml}")
        try:
            document = export.documents_dict[img_filename]
        except KeyError:
            logging.warning(f"{path_img} not labeled, skipping.")
            continue

        pagexml = PageLayout(file=path_xml)
        document.map_to_pagexml(pagexml)
        bites = yolo.produce_bites(path_img, pagexml)
        try:
            graph = graph_provider.get_graph_from_file(bites, path_img, pagexml, document)
        except RuntimeError:
            bad_files += 1
            logging.warning(f"Runtime error detected, skipping. (total {bad_files} bad files)")
            continue

        graphs.append(graph)

    os.makedirs(args.save, exist_ok=True)
    save_path = os.path.join(args.save, args.filename)
    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)


if __name__ == "__main__":
    main()
