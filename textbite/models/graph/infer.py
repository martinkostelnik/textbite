import sys
import os
import argparse
import logging
import pickle

import torch

from safe_gpu import safe_gpu

from textbite.models.graph.model import load_gcn
from textbite.models.graph.create_graphs import Graph  # needed for unpickling

from textbite.models.graph.train import evaluate


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a pickle file with training data.")
    parser.add_argument("--model", required=True, type=str, help="Where to get the model")
    parser.add_argument("--normalizer", type=str, help="Where to get the normalizer")

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        force=True,
    )
    args = parse_arguments()
    logging.info(f'{args}')
    safe_gpu.claim_gpus()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on: {device}")

    with open(args.data, "rb") as f:
        data = pickle.load(f)[:10]
    if args.normalizer:
        with open(args.normalizer, 'rb') as f:
            normalizer = pickle.load(f)
        normalizer.normalize_graphs(data)

    model = load_gcn(args.model, device)
    model.eval()

    logging.info(f'There are {len(data)} graphs')

    logging.info("Evaluating...")

    evaluate(model, data, device)


if __name__ == "__main__":
    main()
