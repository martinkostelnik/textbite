"""Graph Neural Network training.

Date -- 15.05.2024
Author -- Martin Kostelnik
"""

from safe_gpu import safe_gpu
safe_gpu.claim_gpus()

import sys
import os
import argparse
import itertools
import logging
from typing import Tuple, List
from time import perf_counter
import pickle
import copy
from typing import Optional

import torch
from sklearn.metrics import classification_report, precision_score, f1_score
import numpy as np

from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.joiner.graph import Graph  # needed for unpickling
from textbite.models.utils import get_similarities, GraphNormalizer, load_graphs

from clearml import Task


DEFAULT_THRESHOLD = 0.7

def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True, type=str, help="Path to a pickle file with training data.")
    parser.add_argument("--val", required=True, type=str, help="Path to a pickle file with validation data.")

    parser.add_argument("-l", "--layers", type=int, default=3, help="Number of GCN blocks in the model.")
    parser.add_argument("-n", "--hidden-width", type=int, default=512, help="Hidden size of layers.")
    parser.add_argument("-o", "--output-size", type=int, default=512, help="Size of output features.")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help="Dropout probability in the model.")

    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    parser.add_argument("--report-interval", type=int, default=10, help="After how many updates to report")

    parser.add_argument("--patience", type=int, default=10, help="How many evaluations to wait for improvement before stopping.")

    parser.add_argument("--project-name", type=str, default=None, help="ClearML project name")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")

    parser.add_argument("--target", choices=["precision", "f1", "loss"], default="precision", help="Target metric for early stopping")

    args = parser.parse_args()
    return args


def collate_custom_graphs(graphs):
    multi_graph = copy.deepcopy(graphs[0])

    multi_graph.id = None  # We don't guarantee this makes any sense
    multi_graph.node_features = torch.concatenate([g.node_features for g in graphs])  # Shape (n_nodes, n_features)
    multi_graph.line_ids = None  # We don't guarantee this makes any sense

    adjusted_edges_index = []
    offset = 0
    for g in graphs:
        adjusted_edges_index.append(g.edge_index + offset)
        offset += len(g.node_features)
    multi_graph.edge_index = torch.concatenate(adjusted_edges_index, dim=1)  # Shape (2, n_edges)

    multi_graph.edge_attr = torch.concatenate([g.edge_attr for g in graphs])  # Shape (n_edges, n_features), but we have none
    multi_graph.labels = torch.concatenate([g.labels for g in graphs])  # Shape (1, n_edges)

    return multi_graph


class TrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.val_book_losses = []
        self.val_dict_losses = []
        self.val_peri_losses = []
        self.val_book_precisions = []
        self.val_dict_precisions = []
        self.val_peri_precisions = []
        self.combined_precisions = []

        self.__data = [
            self.train_losses,
            self.val_book_losses,
            self.val_dict_losses,
            self.val_peri_losses,
            self.val_book_precisions,
            self.val_dict_precisions,
            self.val_peri_precisions,
            self.combined_precisions,
        ]

    def save(self, path):
        pth = os.path.join(path, "training_data.txt")
        with open(pth, "w") as f:
            for member in self.__data:
                tmp = [str(value) for value in member]
                result_str = " ".join(tmp)
                print(result_str, file=f)


def evaluate(model, data, device, criterion, type: str, target: str):
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
            similarities = (similarities + 1.0) / 2.0
            loss = criterion(similarities, labels)

        val_loss += loss.cpu().item()        
        all_labels.extend(labels.cpu().tolist())
        all_similarities.extend(similarities.cpu().tolist())

    loss = val_loss / len(data)

    if target == "loss":
        best_predictions = [int(similarity >= DEFAULT_THRESHOLD) for similarity in all_similarities]
        precision = precision_score(all_labels, best_predictions)
        print(f"Val loss {type}: {loss:.4f} | Default threshold: {DEFAULT_THRESHOLD:.2f}")
        print(classification_report(all_labels, best_predictions, digits=4))
        model.train()
        return loss, DEFAULT_THRESHOLD, precision
    else:
        best = [0, 0] # [threshold, metric]
        best_predictions = []
        for threshold in np.arange(0.5, 0.99, 0.01):
            predictions = [int(similarity > threshold) for similarity in all_similarities]
            metric = precision_score(all_labels, predictions) if target == "precision" else f1_score(all_labels, predictions)
            if metric > best[1]:
                best = [threshold, metric]
                best_predictions = predictions

        print(f"Val loss {type}: {loss:.4f} | Best threshold: {best[0]:.2f}")
        print(classification_report(all_labels, best_predictions, digits=4))
        model.train()
        return loss, best[0], best[1]


def train(
        model: JoinerGraphModel,
        device,
        train_data,
        val_data,
        report_interval: int,
        lr: float,
        batch_size: int,
        save_path: str,
        patience: int,
        target: str,
        task: Optional[Task]=None,
):
    model.train()

    monitor = TrainingMonitor()

    optim = torch.optim.AdamW(model.parameters(), lr)
    criterion = torch.nn.BCELoss()

    running_loss = 0.0
    best_val_metric = 0.0 if target != "loss" else 9999.0
    best_val_threshold = 0.5
    batch = []
    batch_id = 0
    patience_counter = 0

    graphs_seen = 0
    
    train_similarities = []
    train_labels = []

    best_model_path = os.path.join(save_path, "best-gcn-joiner.pth")
    last_model_path = os.path.join(save_path, "last-gcn-joiner.pth")

    t_start = perf_counter()
    try:
        for graph_i, graph in enumerate(itertools.cycle(train_data)):
            batch.append(graph)
            if len(batch) < batch_size:
                continue

            model.train()
            graph = collate_custom_graphs(batch)
            graphs_seen += len(batch)
            batch = []
            batch_id += 1

            node_features = graph.node_features.to(device)
            edge_indices = graph.edge_index.to(device)
            edge_attrs = graph.edge_attr.to(device)
            labels = graph.labels.to(device, dtype=torch.float32)

            outputs = model(node_features, edge_indices, edge_attrs)
            similarities = get_similarities(outputs, edge_indices)
            similarities = (similarities + 1.0) / 2.0
            train_loss = criterion(similarities, labels)

            optim.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running_loss += train_loss.cpu().item()
            train_similarities.extend(similarities.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

            if batch_id % report_interval == 0:
                t_elapsed_ms = 1000.0 * (perf_counter() - t_start) / report_interval

                train_predictions = [int(similarity > best_val_threshold) for similarity in train_similarities]

                print("TRAIN REPORT:")
                print(f"After {batch_id} Batches ({graph_i + 1} Graphs):")
                print(f"Time {t_elapsed_ms:.1f} ms /B ", end="")
                print(f"({(t_elapsed_ms / batch_size):.1f} ms /G) | ", end="")
                loss = (running_loss / graphs_seen)
                print(f"loss {loss:.4f} /G | {patience_counter=}")
                print(classification_report(train_labels, train_predictions, digits=4))

                train_metric = f1_score(train_labels, train_predictions) if target == "f1" else precision_score(train_labels, train_predictions)

                monitor.train_losses.append(loss)

                graphs_seen = 0
                running_loss = 0.0
                train_similarities = []
                train_labels = []

                print("EVALUATION REPORT:")
                val_loss, val_threshold, val_metric = evaluate(model, val_data, device, criterion, "all", target=target)
                print()

                if task is not None:
                    task.logger.report_scalar(title="Loss", series="train", value=loss, iteration=batch_id)
                    task.logger.report_scalar(title="Loss", series="val", value=val_loss, iteration=batch_id)
                    task.logger.report_scalar(title=target, series="train", value=train_metric, iteration=batch_id)
                    task.logger.report_scalar(title=target, series="val", value=val_metric, iteration=batch_id)

                condition = val_metric > best_val_metric if target != "loss" else val_loss < best_val_metric
                if condition:
                    print(f"Found new best model with {target}: {(val_metric if target != "loss" else val_loss):.4f} (previous: {best_val_metric:.4f}). Saving ...")
                    best_val_metric = val_metric if target != "loss" else val_loss
                    best_val_threshold = val_threshold
                    torch.save(
                        {**model.dict_for_saving, "classification_threshold": best_val_threshold},
                        best_model_path,
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Stopping training due to losing patience. Best model saved at {best_model_path}.")
                        break

                monitor.val_book_losses.append(val_loss)
                monitor.val_book_precisions.append(val_metric)
                monitor.combined_precisions.append(val_metric)

                t_start = perf_counter()

    except KeyboardInterrupt:
        pass

    torch.save(
        {**model.dict_for_saving, "classification_threshold": val_threshold},
        last_model_path,
    )

    if task is not None:
        task.logger.report_single_value(f"Highest validation {target}", best_val_metric)
        task.logger.report_single_value("Best validation threshold", best_val_threshold)

    return monitor


def main():
    logging.basicConfig(
        level=logging.INFO,
        force=True,
    )
    args = parse_arguments()
    logging.info(f'{args}')

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on: {device}")

    # Data
    logging.info("Loading data ...")
    train_graphs, val_graphs = load_graphs(args.train, args.val)
    logging.info("Data loaded.")

    # Normalizer
    logging.info("Creating normalizer ...")
    normalizer = GraphNormalizer(train_graphs)
    logging.info("Normalizer created")

    logging.info("Normalizing train data ...")
    normalizer.normalize_graphs(train_graphs)
    logging.info("Normalizing validation data ...")
    normalizer.normalize_graphs(val_graphs)
    logging.info("Train and validation data normalized.")

    clearml_task = Task.init(project_name=args.project_name, task_name=args.name) if args.project_name is not None else None

    save_path = os.path.join(args.project_name, args.name) if args.project_name is not None else args.name

    # Output folders
    os.makedirs(save_path, exist_ok=True)

    logging.info("Saving normalizer ...")
    with open(os.path.join(save_path, "normalizer.pkl"), 'wb') as f:
        pickle.dump(normalizer, f)
    logging.info("Normalizer created and saved.")

    # Model
    logging.info("Creating model ...")
    input_size = train_graphs[0].node_features.size(dim=1)
    edge_dim = train_graphs[0].edge_attr.size(dim=1)
    model = JoinerGraphModel(
        device=device,
        input_size=input_size,
        edge_dim=edge_dim,
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

    results = train(
        model=model,
        device=device,
        train_data=train_graphs,
        val_data=val_graphs,
        report_interval=args.report_interval,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=save_path,
        patience=args.patience,
        task=clearml_task,
        target=args.target,
    )
    end = perf_counter()
    t = end - start
    logging.info(f"Training finished. Took {t:.1f} s")

    logging.info("Saving training data ...")
    results.save(save_path)
    logging.info("Training data saved.")

    if clearml_task is not None:
        clearml_task.close()


if __name__ == "__main__":
    main()
