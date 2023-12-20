import sys
import argparse
import json
import logging
import os
import pickle

import numpy as np
import torch

from sklearn.cluster import DBSCAN, HDBSCAN

from safe_gpu import safe_gpu

from textbite.models.graph.model import load_gcn, get_similarities
from textbite.models.graph.create_graphs import Graph  # needed for unpickling

from textbite.models.graph.train import evaluate


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a pickle file with training data.")
    parser.add_argument("--model", required=True, type=str, help="Where to get the model")
    parser.add_argument("--normalizer", type=str, help="Where to get the normalizer")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")
    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])

    args = parser.parse_args()
    return args


def bites_from_graph(graph, model):
    node_features = graph.node_features.to(model.device)
    edge_indices = graph.edge_index.to(model.device)

    with torch.no_grad():
        outputs = model(node_features, edge_indices)
        sims = torch.sigmoid(get_similarities(outputs, edge_indices))

    dist_matrix = np.full((len(node_features), len(node_features)), 1)
    for sim, src, dst in zip(sims, edge_indices[0], edge_indices[1]):
        dist_matrix[src, dst] = 1 - sim

    hdbscan = HDBSCAN(metric='precomputed', min_samples=2).fit(dist_matrix)

    bites = [[] for _ in range(max(hdbscan.labels_ + 1))]

    for label, line_id in zip(hdbscan.labels_, graph.line_ids):
        if label == -1:
            continue  # deemed not a member of a cluster, OK
        elif label < 0:
            raise ValueError('Clustering gave unexpected cluster ID {label}')

        bites[label].append(line_id)

    return bites


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)

    logging.info(f'{args}')
    safe_gpu.claim_gpus()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on: {device}")

    with open(args.data, "rb") as f:
        data = pickle.load(f)
    logging.info(f'There are {len(data)} graphs')

    if args.normalizer:
        with open(args.normalizer, 'rb') as f:
            normalizer = pickle.load(f)
        normalizer.normalize_graphs(data)

    model = load_gcn(args.model, device)
    model.eval()

    os.makedirs(args.save, exist_ok=True)

    for graph in data:
        logging.info(f"Processing: {graph.graph_id}")
        result = bites_from_graph(graph, model)

        out_path = os.path.join(args.save, f'{graph.graph_id}.json')
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
