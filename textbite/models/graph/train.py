import sys
import os
import argparse
import itertools
import logging
from typing import Tuple, List
import time
from time import perf_counter
import pickle

import torch

from safe_gpu import safe_gpu

from textbite.models.graph.model import GraphModel, NodeNormalizer
from textbite.models.graph.create_graphs import Graph  # needed for unpickling
from textbite.models.graph.create_graphs import collate_custom_graphs
from textbite.utils import FILENAMES_EXCLUDED_FROM_TRAINING, VALIDATION_FILENAMES_BOOK, VALIDATION_FILENAMES_DICTIONARY, VALIDATION_FILENAMES_PERIODICAL


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a pickle file with training data.")

    parser.add_argument("-l", "--layers", type=int, default=2, help="Number of layers in the model.")
    parser.add_argument("-n", "--hidden-width", type=int, default=128, help="Hidden size of layers.")
    parser.add_argument("-o", "--output-size", type=int, default=128, help="Size of output features.")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help="Dropout probability in the model.")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    parser.add_argument("--report-interval", type=int, default=10, help="After how many updates to report")
    parser.add_argument("--save", type=str, help="Where to save the model best by validation F1")
    parser.add_argument("--checkpoint-dir", type=str, help="Where to put all training checkpoints")

    args = parser.parse_args()
    return args


def load_data(path: str) -> Tuple[List[Graph], List[Graph], List[Graph], List[Graph]]:
    start = time.perf_counter()
    with open(path, "rb") as f:
        data = pickle.load(f)

    train_data = [graph for graph in data if f"{graph.graph_id}.jpg" not in FILENAMES_EXCLUDED_FROM_TRAINING]
    val_data_book = [graph for graph in data if f"{graph.graph_id}.jpg" in VALIDATION_FILENAMES_BOOK]
    val_data_dict = [graph for graph in data if f"{graph.graph_id}.jpg" in VALIDATION_FILENAMES_DICTIONARY]
    val_data_peri = [graph for graph in data if f"{graph.graph_id}.jpg" in VALIDATION_FILENAMES_PERIODICAL]

    end = time.perf_counter()
    logging.info(f"Train graphs: {len(train_data)} | Val graphs book: {len(val_data_book)} | Val graphs dictionary: {len(val_data_dict)} | Val graphs periodical: {len(val_data_peri)} | Took: {(end-start):.3f} s")

    return train_data, val_data_book, val_data_dict, val_data_peri


def get_similarities(node_features, edge_indices):
    lhs_nodes = torch.index_select(input=node_features, dim=0, index=edge_indices[0, :])
    rhs_nodes = torch.index_select(input=node_features, dim=0, index=edge_indices[1, :])
    fea_dim = lhs_nodes.shape[1]
    similarities = torch.sum(lhs_nodes * rhs_nodes / fea_dim, dim=1)

    return similarities


# TODO: DET or similar will be super useful, it's all terribly mis-calibrated so far
def per_edge_accuracy(similarities, labels, threshold: float=0.5):
    return torch.sum((similarities > threshold) == labels) / len(labels)


def evaluate(model, data, device, criterion, type: str):
    model.eval()

    val_loss = 0.0
    accuracy = 0.0
    for graph in data:
        node_features = graph.node_features.to(device)
        edge_indices = graph.edge_index.to(device)
        labels = graph.labels.to(device, dtype=torch.float32)
        edge_attrs = graph.edge_attr.to(device)

        with torch.no_grad():
            outputs = model(node_features, edge_indices, edge_attrs)
            similarities = get_similarities(outputs, edge_indices)
            loss = criterion(similarities, labels)
            accuracy += per_edge_accuracy(similarities, labels)
        val_loss += loss.cpu().item()

    print(f'Val loss {type}: {val_loss/len(data):.4f} Edge accuracy: {100.0 * accuracy/len(data):.2f} %')

    return val_loss


def train(
        model: GraphModel,
        device,
        train_data,
        val_data_book,
        val_data_dict,
        val_data_peri,
        report_interval: int,
        lr: float,
        batch_size: int,
        save_path: str,
        checkpoint_dir: str,
):
    optim = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    # criterion = torch.nn.BCEWithLogitsLoss()

    model.train()

    accs = []
    losses = []
    running_loss = 0.0
    t0 = time.time()
    acc = 0.0

    batch = []
    batch_id = 0

    try:
        for graph_i, graph in enumerate(itertools.cycle(train_data)):
            batch.append(graph)
            if len(batch) < batch_size:
                continue

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
            acc += per_edge_accuracy(similarities, labels)  # Note that this is distorted now, weight of graphs depends on who they meet in a batch

            optim.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running_loss += train_loss.cpu().item()

            if batch_id % report_interval == 0:
                t_diff = time.time() - t0
                print(f"After {batch_id} Batches ({batch_id*batch_size} Graphs): time {1000.0*t_diff/(report_interval):.1f}ms /B ({1000.0*t_diff/(report_interval*batch_size):.1f}ms /G), loss {running_loss/(report_interval):.4f} /B {running_loss/(report_interval*batch_size):.4f} /G, acc {100.0*acc/(report_interval):.2f} %")
                accs.append(100.0 * acc.cpu().item()/report_interval)
                losses.append(running_loss/report_interval)
                running_loss = 0.0
                acc = 0.0
                t0 = time.time()

                evaluate(model, val_data_book, device, criterion, "book")
                evaluate(model, val_data_dict, device, criterion, "dictionary")
                evaluate(model, val_data_peri, device, criterion, "periodical")
                print()

                if checkpoint_dir:
                    torch.save(model.dict_for_saving, os.path.join(checkpoint_dir, f"{model.__class__.__name__}-checkpoint.{graph_i}.pth"))

            # if f1_val > best_f1_val:
            #     best_f1_val = f1_val
            #     print(f"SAVING MODEL at f1_val = {f1_val}")
            #     torch.save(dict_for_saving, os.path.join(save_path, "BaselineModel.pth"))

            # print(f"Epoch {epoch + 1} finished | train f1 = {f1:.2f} loss = {train_loss/train_nb_examples:.3e} | val f1 = {f1_val:.2f} loss = {val_loss/val_nb_examples:.3e} {'| Only zeros in last val batch!' if not any(preds) else ''}")
            # if (epoch + 1) % 10 == 0:
            #     target_names = ["None", "Terminating", "Title"]  # should be provided by model or someone like that
            #     print("TRAIN REPORT:")
            #     print(classification_report(epoch_labels, epoch_preds, digits=4, zero_division=0, target_names=target_names))
            #     print(confusion_report(epoch_labels, epoch_preds, target_names))
            #     print("VALIDATION REPORT:")
            #     print(classification_report(epoch_labels_val, epoch_preds_val, digits=4, zero_division=0, target_names=target_names))
            #     print(confusion_report(epoch_labels_val, epoch_preds_val, target_names))
            #     print("VALIDATION REPORT:")
    except KeyboardInterrupt:
        pass

    return accs, losses


def main():
    logging.basicConfig(
        level=logging.INFO,
        force=True,
    )
    args = parse_arguments()
    logging.info(f'{args}')
    safe_gpu.claim_gpus()

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on: {device}")

    logging.info("Loading data ...")
    train_graphs, val_graphs_book, val_graphs_dict, val_graphs_peri = load_data(args.data)
    # train_graphs = train_graphs[:50]
    logging.info(f"Data loaded.")

    logging.info("Creating normalizer ...")
    normalizer = NodeNormalizer(train_graphs)
    logging.info("Saving normalizer ...")
    os.makedirs(args.save, exist_ok=True)
    with open(os.path.join(args.save, "normalizer.pkl"), 'wb') as f:
        pickle.dump(normalizer, f)
    logging.info("Normalizer created and saved.")

    logging.info("Normalizing train data ...")
    normalizer.normalize_graphs(train_graphs)
    logging.info("Normalizing validation data ...")
    normalizer.normalize_graphs(val_graphs_book)
    normalizer.normalize_graphs(val_graphs_dict)
    normalizer.normalize_graphs(val_graphs_peri)
    logging.info("Train and validation data normalized.")

    # Model
    logging.info("Creating model ...")
    input_size = train_graphs[0].node_features.size(dim=1)
    model = GraphModel(
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

    accs, losses = train(
        model=model,
        device=device,
        train_data=train_graphs,
        val_data_book=val_graphs_book,
        val_data_dict=val_graphs_dict,
        val_data_peri=val_graphs_peri,
        report_interval=args.report_interval,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save,
        checkpoint_dir=args.checkpoint_dir,
    )
    end = perf_counter()
    t = end - start
    logging.info(f"Training finished. Took {t:.1f} s")

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.plot(accs)
    plt.subplot(1, 2, 2)
    plt.plot(losses)

    plt.savefig("asdf.pdf")

if __name__ == "__main__":
    main()
