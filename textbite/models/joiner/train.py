import sys
import os
import argparse
import itertools
import logging
from typing import Tuple, List
from time import perf_counter
import pickle

import torch
from sklearn.metrics import classification_report

from safe_gpu import safe_gpu

from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.joiner.graph import Graph  # needed for unpickling
from textbite.models.graph.model import NodeNormalizer, EdgeNormalizer
from textbite.models.graph.create_graphs import collate_custom_graphs
from textbite.models.utils import get_similarities


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True, type=str, help="Path to a pickle file with training data.")
    parser.add_argument("--val-book", required=True, type=str, help="Path to a pickle file with validation book data.")
    parser.add_argument("--val-dict", required=True, type=str, help="Path to a pickle file with validation dict data.")
    parser.add_argument("--val-peri", required=True, type=str, help="Path to a pickle file with validation periodical data.")

    parser.add_argument("-l", "--layers", type=int, default=2, help="Number of GCN blocks in the model.")
    parser.add_argument("-n", "--hidden-width", type=int, default=128, help="Hidden size of layers.")
    parser.add_argument("-o", "--output-size", type=int, default=128, help="Size of output features.")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help="Dropout probability in the model.")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Classification threshold.")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    parser.add_argument("--report-interval", type=int, default=10, help="After how many updates to report")
    parser.add_argument("--save", type=str, help="Where to save the model best by validation F1")
    parser.add_argument("--checkpoint-dir", type=str, help="Where to put all training checkpoints")

    args = parser.parse_args()
    return args


def load_data(
    path_train: str,
    path_val_book: str,
    path_val_dict: str,
    path_val_peri: str,
) -> Tuple[List[Graph], List[Graph], List[Graph], List[Graph]]:
    start = perf_counter()
    with open(path_train, "rb") as f:
        train_data = pickle.load(f)

    with open(path_val_book, "rb") as f:
        val_data_book = pickle.load(f)

    with open(path_val_dict, "rb") as f:
        val_data_dict = pickle.load(f)

    with open(path_val_peri, "rb") as f:
        val_data_peri = pickle.load(f)

    end = perf_counter()
    logging.info(f"Train graphs: {len(train_data)} | Val graphs book: {len(val_data_book)} | Val graphs dictionary: {len(val_data_dict)} | Val graphs periodical: {len(val_data_peri)} | Took: {(end-start):.3f} s")

    return train_data, val_data_book, val_data_dict, val_data_peri


def evaluate(model, data, device, criterion, type: str, threshold: float):
    model.eval()

    val_loss = 0.0
    all_similarities = []
    all_labels = []

    for graph in data:
        node_features = graph.node_features.to(device)
        edge_indices = graph.edge_index.to(device)
        labels = graph.labels.to(device, dtype=torch.float32)
        edge_attrs = graph.edge_attr.to(device)

        with torch.no_grad():
            outputs = model(node_features, edge_indices, edge_attrs)
            similarities = get_similarities(outputs, edge_indices)
            loss = criterion(similarities, labels)

        val_loss += loss.cpu().item()        
        all_labels.extend(labels.cpu().tolist())
        all_similarities.extend(similarities.cpu().tolist())

    all_predictions = [int(similarity > threshold) for similarity in all_similarities]

    print(f"Val loss {type}: {val_loss/len(data):.4f}")
    print(classification_report(all_labels, all_predictions, digits=4))

    return val_loss


def train(
        model: JoinerGraphModel,
        device,
        train_data,
        val_data_book,
        val_data_dict,
        val_data_peri,
        report_interval: int,
        lr: float,
        threshold: float,
        batch_size: int,
        save_path: str,
        checkpoint_dir: str,
):
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr)
    # criterion = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor([0.01]).to(device))
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCELoss(reduction="sum")

    running_loss = 0.0
    batch = []
    batch_id = 0

    all_similarities = []
    all_labels = []

    t_start = perf_counter()
    try:
        for graph_i, graph in enumerate(itertools.cycle(train_data)):
            batch.append(graph)
            if len(batch) < batch_size:
                continue

            model.train()
            graph = collate_custom_graphs(batch)
            batch = []
            batch_id += 1

            node_features = graph.node_features.to(device)
            edge_indices = graph.edge_index.to(device)
            edge_attrs = graph.edge_attr.to(device)
            labels = graph.labels.to(device, dtype=torch.float32)

            outputs = model(node_features, edge_indices, edge_attrs)
            similarities = get_similarities(outputs, edge_indices)
            train_loss = criterion(similarities, labels)

            optim.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running_loss += train_loss.cpu().item()
            all_similarities.extend(similarities.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if batch_id % report_interval == 0:
                t_elapsed_ms = 1000.0 * (perf_counter() - t_start) / report_interval

                all_predictions = [int(similarity > threshold) for similarity in all_similarities]

                print("TRAIN REPORT:")
                print(f"After {batch_id} Batches ({graph_i + 1} Graphs):")
                print(f"Time {t_elapsed_ms:.1f} ms /B ", end="")
                print(f"({(t_elapsed_ms / batch_size):.1f} ms /G) | ", end="")
                print(f"loss {running_loss / (report_interval * batch_size):.4f} /G")
                print(classification_report(all_labels, all_predictions, digits=4))

                running_loss = 0.0
                all_similarities = []
                all_labels = []

                print("EVALUATION REPORT:")
                evaluate(model, val_data_book, device, criterion, "book", threshold)
                evaluate(model, val_data_dict, device, criterion, "dictionary", threshold)
                evaluate(model, val_data_peri, device, criterion, "periodical", threshold)
                print()
                model.train()

                if checkpoint_dir:
                    checkpoint_filename = f"{model.__class__.__name__}-joiner-checkpoint.{graph_i}.pth"
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                    torch.save(
                        model.dict_for_saving,
                        checkpoint_path,
                    )

                t_start = perf_counter()

    except KeyboardInterrupt:
        pass


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

    # Data
    logging.info("Loading data ...")
    train_graphs, val_graphs_book, val_graphs_dict, val_graphs_peri = load_data(args.train, args.val_book, args.val_dict, args.val_peri)
    logging.info(f"Data loaded.")

    logging.info("Creating normalizer ...")
    normalizer = NodeNormalizer(train_graphs)
    logging.info("Normalizer created")

    logging.info("Normalizing train data ...")
    normalizer.normalize_graphs(train_graphs)
    logging.info("Normalizing validation data ...")
    normalizer.normalize_graphs(val_graphs_book)
    normalizer.normalize_graphs(val_graphs_dict)
    normalizer.normalize_graphs(val_graphs_peri)
    logging.info("Train and validation data normalized.")

    logging.info("Creating normalizer ...")
    edge_normalizer = EdgeNormalizer(train_graphs)
    logging.info("Normalizer created")

    logging.info("Normalizing train data ...")
    edge_normalizer.normalize_graphs(train_graphs)
    logging.info("Normalizing validation data ...")
    edge_normalizer.normalize_graphs(val_graphs_book)
    edge_normalizer.normalize_graphs(val_graphs_dict)
    edge_normalizer.normalize_graphs(val_graphs_peri)
    logging.info("Train and validation data normalized.")

    # Output folders
    os.makedirs(args.save, exist_ok=True)
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    logging.info("Saving normalizer ...")
    with open(os.path.join(args.save, "normalizer.pkl"), 'wb') as f:
        pickle.dump(normalizer, f)
    logging.info("Normalizer created and saved.")

    # Model
    logging.info("Creating model ...")
    input_size = train_graphs[0].node_features.size(dim=1)
    model = JoinerGraphModel(
        device=device,
        input_size=input_size,
        output_size=args.output_size,
        n_layers=args.layers,
        hidden_size=args.hidden_width,
        dropout_prob=args.dropout,
    )
    print(model)
    model = model.to(device)
    logging.info("Model created.")

    logging.info("Starting training ...")
    start = perf_counter()

    train(
        model=model,
        device=device,
        train_data=train_graphs,
        val_data_book=val_graphs_book,
        val_data_dict=val_graphs_dict,
        val_data_peri=val_graphs_peri,
        report_interval=args.report_interval,
        lr=args.lr,
        threshold=args.threshold,
        batch_size=args.batch_size,
        save_path=args.save,
        checkpoint_dir=args.checkpoint_dir,
    )
    end = perf_counter()
    t = end - start
    logging.info(f"Training finished. Took {t:.1f} s")


if __name__ == "__main__":
    main()
