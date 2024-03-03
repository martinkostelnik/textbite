#!/usr/bin/env python3

import argparse
import logging
import os.path

from ultralytics import YOLO
from safe_gpu import safe_gpu
import torch

from pero_ocr.document_ocr.layout import PageLayout

from textbite.models.yolo.infer import YoloBiter
from textbite.models.joiner.graph import JoinerGraphProvider
from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.joiner.train import get_similarities


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xmls", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--yolo", required=True, type=str, help="Path to the .pt file with weights of YOLO model.")
    parser.add_argument("--model", required=True, type=str, help="Path to the .pt file with weights of Joiner model.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    safe_gpu.claim_gpus()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on: {device}")

    logging.info("Loading model checkpoint ...")
    model_checkpoint = torch.load(args.model)
    logging.info("Model checkpoint loaded.")

    logging.info("Creating YOLO biter model ...")
    biter = YoloBiter(YOLO(args.yolo))
    logging.info("YOLO biter model created.")

    logging.info("Creating Graph provider ...")
    graph_provider = JoinerGraphProvider()
    logging.info("Graph provider created.")

    logging.info("Creating Joiner model ...")
    joiner = JoinerGraphModel(
        device=device,
        input_size=model_checkpoint["input_size"],
        output_size=model_checkpoint["output_size"],
        n_layers=model_checkpoint["n_layers"],
        hidden_size=model_checkpoint["hidden_size"],
    )
    joiner = joiner.to(device)
    logging.info("Joiner model created.")

    logging.info("Loading weights ...")
    joiner.load_state_dict(model_checkpoint["state_dict"])
    logging.info("Weights loaded.")

    logging.info("Creating save directory ...")
    os.makedirs(args.save, exist_ok=True)
    logging.info("Save directory created.")

    xml_filenames = [filename for filename in os.listdir(args.xmls) if filename.endswith(".xml")]

    logging.info("Starting inference ...")
    for i, xml_filename in enumerate(xml_filenames):
        img_filename = xml_filename.replace(".xml", ".jpg")
        path_xml = os.path.join(args.xmls, xml_filename)
        path_img = os.path.join(args.images, img_filename)
        logging.info(f"({i}/{len(xml_filenames)}) | Processing: {path_xml}")

        pagexml = PageLayout(file=path_xml)
        try:
            graph = graph_provider.get_graph_from_file(biter, path_img, pagexml)
        except RuntimeError:
            logging.info(f"Single region detected on {img_filename}, skipping.")
            continue

        node_features = graph.node_features.to(device)
        edge_indices = graph.edge_index.to(device)
        edge_attrs = graph.edge_attr.to(device)

        outputs = joiner(node_features, edge_indices, edge_attrs)
        similarities = get_similarities(outputs, edge_indices)

        positive_predictions = torch.sum(similarities > 0.5)
        print(positive_predictions)
    
        # out_path = os.path.join(args.save, filename.replace(".xml", ".json"))
        # save_result(result, out_path)


if __name__ == '__main__':
    main()
