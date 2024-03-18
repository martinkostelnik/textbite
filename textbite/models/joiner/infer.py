#!/usr/bin/env python3

import argparse
import logging
import os.path
from typing import List, Tuple, Dict
import json
import pickle

from safe_gpu import safe_gpu
import torch
from transformers import BertTokenizerFast, BertModel

from pero_ocr.document_ocr.layout import PageLayout

from textbite.models.joiner.graph import JoinerGraphProvider, Graph
from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.MLP import MLP
from textbite.models.autoencoder import AutoEncoder
from textbite.models.utils import edge_indices_to_edges, get_transitive_subsets, GraphNormalizer, get_similarities, ModelType
from textbite.utils import CZERT_PATH
from textbite.bite import load_bites, Bite, save_bites


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--data", required=True, type=str, help="Path to a folder with jsons containing bites.")
    parser.add_argument("--xmls", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--model", required=True, type=str, help="Path to the .pt file with weights of Joiner model.")
    parser.add_argument("--normalizer", required=True, type=str, help="Path to node normalizer.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")

    return parser.parse_args()


def join_bites_by_dict(nodes_to_join: Dict[int, List[int]], bites: List[Bite]) -> List[Bite]:
    to_delete = []
    for kept_bite_idx, bites_indices in nodes_to_join.items():
        to_delete.extend(bites_indices)
        for bite_index in bites_indices:
            bites[kept_bite_idx].lines.extend(bites[bite_index].lines)

    bites_new = [bite for i, bite in enumerate(bites) if i not in to_delete]

    return bites_new


def get_joining_dict(positive_edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    subsets = get_transitive_subsets(positive_edges)

    nodes_to_join = []
    for subset in subsets:
        nodes = set()
        for edge in subset:
            nodes.update(edge)
        nodes_to_join.append(nodes)

    nodes_to_join = [list(ss) for ss in nodes_to_join]
    nodes_to_join = {s[0]: s[1:] for s in nodes_to_join}

    return nodes_to_join


def get_positive_edges_gcn(
        graph: Graph,
        joiner: MLP,
        device,
        threshold: float,
    ) -> List[Bite]:
    node_features = graph.node_features.to(device)
    edge_indices = graph.edge_index.to(device)
    edge_attrs = graph.edge_attr.to(device)

    with torch.no_grad():
        outputs = joiner(node_features, edge_indices, edge_attrs)
    similarities = get_similarities(outputs, edge_indices)
    similarities = similarities.tolist()
    positive_edge_indices = [index for index, similarity in enumerate(similarities) if similarity >= threshold]

    edge_indices = edge_indices.detach().cpu().tolist()
    from_indices, to_indices = edge_indices[0], edge_indices[1]
    edges = edge_indices_to_edges(from_indices, to_indices)

    positive_edges = [edge for idx, edge in enumerate(edges) if idx in positive_edge_indices]
    return positive_edges


def get_positive_edges_mlp(
        graph: Graph,
        joiner: MLP,
        device,
        threshold: float,
    ) -> List[Bite]:
    graph_features, _, edges = graph.flatten()

    positive_edges = []

    for edge_features, edge in zip(graph_features, edges):
        edge_features = edge_features.to(device)

        with torch.no_grad():
            logits = joiner(edge_features).squeeze()

        probability = torch.sigmoid(logits).cpu().item()
        if probability > threshold:
            positive_edges.append(edge)

    return positive_edges


def get_positive_edges_ae(
        graph: Graph,
        joiner: AutoEncoder,
        device,
        threshold: float,
    ) -> List[Bite]:
    graph_features, _, edges = graph.flatten()

    positive_edges = []

    for edge_features, edge in zip(graph_features, edges):
        edge_features = edge_features.to(device)

        with torch.no_grad():
            _, reconstructed_features = joiner(edge_features)
            reconstruction_error = torch.nn.functional.mse_loss(reconstructed_features, edge_features).cpu().item()
            
        if reconstruction_error > threshold:
            positive_edges.append(edge)

    return positive_edges


def join_bites(
        bites: List[Bite],
        joiner: JoinerGraphModel,
        graph_provider: JoinerGraphProvider,
        normalizer: GraphNormalizer,
        filename: str,
        pagexml: PageLayout,
        device,
        threshold: float
        ) -> List[Bite]:
    graph = graph_provider.get_graph_from_bites(bites, filename, pagexml)

    normalizer.normalize_graphs([graph])

    match joiner.model_type:
        case ModelType.MLP:
            positive_edges = get_positive_edges_mlp(graph, joiner, device, threshold)

        case ModelType.GCN:
            positive_edges = get_positive_edges_gcn(graph, joiner, device, threshold)

        case ModelType.AE:
            positive_edges = get_positive_edges_ae(graph, joiner, device, threshold)

    nodes_to_join = get_joining_dict(positive_edges)
    new_bites = join_bites_by_dict(nodes_to_join, bites)

    return new_bites


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    safe_gpu.claim_gpus()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Inference running on {device}")

    logging.info("Loading model checkpoint ...")
    model_checkpoint = torch.load(args.model)
    threshold = model_checkpoint["classification_threshold"] if "classification_threshold" in model_checkpoint.keys() else 0.5
    logging.info("Model checkpoint loaded.")

    logging.info("Loading language model ...")
    tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
    logging.info("Tokenizer loaded.")
    logging.info("Loading CZERT ...")
    czert = BertModel.from_pretrained(CZERT_PATH)
    czert = czert.to(device)
    logging.info("Language model loaded.")

    logging.info("Creating Graph provider ...")
    graph_provider = JoinerGraphProvider(tokenizer, czert, device)
    logging.info("Graph provider created.")

    logging.info("Creating normalizer ...")
    with open(args.normalizer, "rb") as f:
        normalizer: GraphNormalizer = pickle.load(f)
    logging.info("Normalizer created.")

    logging.info("Creating Joiner model ...")
    match model_checkpoint["model_type"]:
        case ModelType.MLP:
            joiner = MLP.from_pretrained(model_checkpoint, device)

        case ModelType.GCN:
            joiner = JoinerGraphModel.from_pretrained(model_checkpoint, device)

        case ModelType.AE:
            joiner = AutoEncoder.from_pretrained(model_checkpoint, device)

        case _:
            raise ValueError("Invalid type of model loaded from checkpoint.")
        
    joiner.eval()
    joiner = joiner.to(device)
    logging.info("Joiner model created.")

    logging.info("Creating save directory ...")
    os.makedirs(args.save, exist_ok=True)
    logging.info("Save directory created.")

    json_filenames = [filename for filename in os.listdir(args.data) if filename.endswith(".json")]

    logging.info("Starting inference ...")
    for i, json_filename in enumerate(json_filenames):
        xml_filename = json_filename.replace(".json", ".xml")
        filename = json_filename.replace(".json", "")

        path_json = os.path.join(args.data, json_filename)
        path_xml = os.path.join(args.xmls, xml_filename)
        save_path = os.path.join(args.save, json_filename)

        logging.info(f"({i}/{len(json_filenames)}) | Processing: {path_json}")

        pagexml = PageLayout(file=path_xml)
        original_bites = load_bites(path_json)

        try:
            new_bites = join_bites(
                original_bites,
                joiner,
                graph_provider,
                normalizer,
                filename,
                pagexml,
                device,
                threshold,
            )
        except RuntimeError:
            logging.info(f"Single region detected on {xml_filename}, saving as is.")
            new_bites = original_bites

        save_bites(new_bites, save_path)


if __name__ == '__main__':
    main()
