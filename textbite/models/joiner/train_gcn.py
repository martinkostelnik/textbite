import sys
import os
import argparse
import itertools
import logging
from typing import Tuple, List
from time import perf_counter
import pickle

import torch
from sklearn.metrics import classification_report, precision_score

from safe_gpu import safe_gpu

from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.joiner.graph import Graph  # needed for unpickling
from textbite.models.graph.create_graphs import collate_custom_graphs
from textbite.models.utils import get_similarities, GraphNormalizer, load_graphs


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
    parser.add_argument("--train_result_path", default=None, type=str, help="Where to save training results.")

    args = parser.parse_args()
    return args


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
        with open(path, "w") as f:
            for member in self.__data:
                tmp = [str(value) for value in member]
                result_str = " ".join(tmp)
                print(result_str, file=f)


def evaluate(model, data, device, criterion, type: str, threshold: float) -> Tuple[float, List[int], List[int]]:
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

    return val_loss, all_predictions, all_labels


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
):
    model.train()

    monitor = TrainingMonitor()

    optim = torch.optim.AdamW(model.parameters(), lr)
    # criterion = torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor([0.01]).to(device))
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCELoss()

    running_loss = 0.0
    best_precision_sum = 0.0
    batch = []
    batch_id = 0

    all_similarities = []
    all_labels = []

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
                loss = (running_loss / report_interval)
                print(f"loss {loss:.4f} /G")
                print(classification_report(all_labels, all_predictions, digits=4))

                monitor.train_losses.append(loss)

                running_loss = 0.0
                all_similarities = []
                all_labels = []

                print("EVALUATION REPORT:")
                val_book_loss, book_predictions, book_labels = evaluate(model, val_data_book, device, criterion, "book", threshold)
                val_dict_loss, dict_predictions, dict_labels = evaluate(model, val_data_dict, device, criterion, "dictionary", threshold)
                val_peri_loss, peri_predictions, peri_labels = evaluate(model, val_data_peri, device, criterion, "periodical", threshold)
                print()
                model.train()

                book_positive_precision = precision_score(book_labels, book_predictions)
                dict_positive_precision = precision_score(dict_labels, dict_predictions)
                peri_positive_precision = precision_score(peri_labels, peri_predictions)
                precision_sum = book_positive_precision + dict_positive_precision + peri_positive_precision
                if precision_sum > best_precision_sum:
                    print(f"Found new best model with combined precision = {precision_sum:.4f} / 3.0. Saving ...")
                    best_precision_sum = precision_sum
                    torch.save(
                        {**model.dict_for_saving, "classification_threshold": threshold},
                        best_model_path,
                    )

                monitor.val_book_losses.append(val_book_loss)
                monitor.val_dict_losses.append(val_dict_loss)
                monitor.val_peri_losses.append(val_peri_loss)
                monitor.val_book_precisions.append(book_positive_precision)
                monitor.val_dict_precisions.append(dict_positive_precision)
                monitor.val_peri_precisions.append(peri_positive_precision)
                monitor.combined_precisions.append(precision_sum)

                t_start = perf_counter()

    except KeyboardInterrupt:
        pass

    torch.save(
        {**model.dict_for_saving, "classification_threshold": threshold},
        last_model_path,
    )

    return monitor


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
    train_graphs, val_graphs_book, val_graphs_dict, val_graphs_peri = load_graphs(args.train, args.val_book, args.val_dict, args.val_peri)
    logging.info(f"Data loaded.")

    # Normalizer
    logging.info("Creating normalizer ...")
    normalizer = GraphNormalizer(train_graphs)
    logging.info("Normalizer created")

    logging.info("Normalizing train data ...")
    normalizer.normalize_graphs(train_graphs)
    logging.info("Normalizing validation data ...")
    normalizer.normalize_graphs(val_graphs_book)
    normalizer.normalize_graphs(val_graphs_dict)
    normalizer.normalize_graphs(val_graphs_peri)
    logging.info("Train and validation data normalized.")

    # Output folders
    os.makedirs(args.save, exist_ok=True)

    logging.info("Saving normalizer ...")
    with open(os.path.join(args.save, "normalizer.pkl"), 'wb') as f:
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
        val_data_book=val_graphs_book,
        val_data_dict=val_graphs_dict,
        val_data_peri=val_graphs_peri,
        report_interval=args.report_interval,
        lr=args.lr,
        threshold=args.threshold,
        batch_size=args.batch_size,
        save_path=args.save,
    )
    end = perf_counter()
    t = end - start
    logging.info(f"Training finished. Took {t:.1f} s")

    if args.train_result_path is not None:
        logging.info("Saving training data ...")
        results.save(args.train_result_path)
        logging.info("Training data saved.")


if __name__ == "__main__":
    main()
