#!/usr/bin/env python3

import argparse
import logging
import os.path
from typing import List
import json
import pickle

from ultralytics import YOLO
from safe_gpu import safe_gpu
import torch

from pero_ocr.document_ocr.layout import PageLayout

from textbite.models.yolo.infer import YoloBiter, Bite
from textbite.models.joiner.graph import JoinerGraphProvider
from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.graph.model import NodeNormalizer
from textbite.models.joiner.train import get_similarities
from textbite.models.utils import edge_indices_to_edges, get_transitive_subsets


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xmls", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--yolo", required=True, type=str, help="Path to the .pt file with weights of YOLO model.")
    parser.add_argument("--model", required=True, type=str, help="Path to the .pt file with weights of Joiner model.")
    parser.add_argument("--normalizer", required=True, type=str, help="Path to node normalizer.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")

    return parser.parse_args()


def join_bites(
        bites: List[Bite],
        joiner: JoinerGraphModel,
        graph_provider: JoinerGraphProvider,
        normalizer: NodeNormalizer,
        path_img: str,
        pagexml: PageLayout,
        device,
        ) -> List[Bite]:
    graph = graph_provider.get_graph_from_file(bites, path_img, pagexml)
    normalizer.normalize_graphs([graph])
    node_features = graph.node_features.to(device)
    edge_indices = graph.edge_index.to(device)
    edge_attrs = graph.edge_attr.to(device)

    with torch.no_grad():
        outputs = joiner(node_features, edge_indices, edge_attrs)
    similarities = get_similarities(outputs, edge_indices)
    similarities = similarities.tolist()
    positive_edge_indices = [index for index, similarity in enumerate(similarities) if similarity >= 0.5]

    edge_indices = edge_indices.detach().cpu().tolist()
    from_indices, to_indices = edge_indices[0], edge_indices[1]
    edges = edge_indices_to_edges(from_indices, to_indices)
    positive_edges = [edge for idx, edge in enumerate(edges) if idx in positive_edge_indices]

    print(f"{len(positive_edges)}/{len(edges)}")
    subsets = get_transitive_subsets(positive_edges)

    return []


def save_result(result: List[Bite], path: str) -> None:
    with open(path, "w") as f:
        json.dump([bite.__dict__ for bite in result], f, indent=4, ensure_ascii=False)


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

    logging.info("Creating node normalizer ...")
    with open(args.normalizer, "rb") as f:
        normalizer = pickle.load(f)
    logging.info("Node normalizer created.")

    logging.info("Creating Joiner model ...")
    joiner = JoinerGraphModel(
        device=device,
        input_size=model_checkpoint["input_size"],
        output_size=model_checkpoint["output_size"],
        n_layers=model_checkpoint["n_layers"],
        hidden_size=model_checkpoint["hidden_size"],
    )
    joiner.eval()
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
        json_filename = xml_filename.replace(".xml", ".json")
        path_xml = os.path.join(args.xmls, xml_filename)
        path_img = os.path.join(args.images, img_filename)
        logging.info(f"({i}/{len(xml_filenames)}) | Processing: {path_xml}")

        pagexml = PageLayout(file=path_xml)
        with torch.no_grad():
            original_bites = biter.produce_bites(path_img, pagexml)

        try:
            new_bites = join_bites(original_bites, joiner, graph_provider, normalizer, path_img, pagexml, device)
        except RuntimeError:
            logging.info(f"Single region detected on {img_filename}, skipping.")
            continue
    
        out_path = os.path.join(args.save, json_filename)
        save_result(new_bites, out_path)


if __name__ == '__main__':
    main()
