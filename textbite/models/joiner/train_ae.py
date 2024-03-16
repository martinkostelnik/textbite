import sys
import argparse
import logging
import os
from time import perf_counter
import pickle
from typing import List, Tuple

import torch
from safe_gpu import safe_gpu
from sklearn.metrics import classification_report, precision_score

from textbite.models.utils import GraphNormalizer, load_graphs
from textbite.models.autoencoder import AutoEncoder
from textbite.models.joiner.graph import Graph


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True, type=str, help="Path to a pickle file with training data.")
    parser.add_argument("--val-book", required=True, type=str, help="Path to a pickle file with validation book data.")
    parser.add_argument("--val-dict", required=True, type=str, help="Path to a pickle file with validation dict data.")
    parser.add_argument("--val-peri", required=True, type=str, help="Path to a pickle file with validation periodical data.")

    parser.add_argument("-n", "--hidden-width", type=int, default=128, help="Hidden size of layers.")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help="Dropout probability in the model.")
    parser.add_argument("--encoding-size", type=int, default=16, help="Latent space size.")

    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    parser.add_argument("--save", type=str, help="Where to save the model best by validation F1")

    args = parser.parse_args()
    return args


class Dataset(torch.utils.data.Dataset):
    def __init__(self, graphs: List[Graph]):
        self.data = []
        self.labels = []
        for graph in graphs:
            edges, labels, _ = graph.flatten()
            self.data.extend(edges)
            self.labels.extend(labels)

        assert len(self.data) == len(self.labels)

        self.positive_data = [sample for sample, label in zip(self.data, self.labels) if label == 1.0]
        self.negative_data = [sample for sample, label in zip(self.data, self.labels) if label == 0.0]

    def __len__(self):
        return len(self.negative_data)

    def __getitem__(self, index: int):
        return self.negative_data[index]
    

def evaluate(
        model: AutoEncoder,
        data: List[Graph],
        device,
        criterion,
        type: str,
        threshold: float,
    ) -> Tuple[float, List[int], List[int]]:
    model.eval()

    val_loss = 0.0
    all_predictions = []
    all_labels = []

    for graph in data:
        graph_features, graph_labels, _ = graph.flatten()

        for edge_features, label in zip(graph_features, graph_labels):
            edge_features = edge_features.to(device)
            label = torch.tensor(label, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, reconstructed = model(edge_features)
                loss = criterion(reconstructed, edge_features).cpu().item()

            val_loss += loss

            all_predictions.append(float(loss > threshold))
            all_labels.append(label.cpu().item())

    print(f"Val loss {type}: {val_loss/len(all_labels):.4f}")
    print(classification_report(all_labels, all_predictions, digits=4))

    return val_loss, all_predictions, all_labels


def train(
        model: AutoEncoder,
        device,
        train_dataset,
        val_graphs_book: List[Graph],
        val_graphs_dict: List[Graph],
        val_graphs_peri: List[Graph],
        batch_size: int,
        epochs: int,
        lr: float,
        save_path: str,
    ):
    model.train()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    optim = torch.optim.AdamW(model.parameters(), lr)
    criterion = torch.nn.MSELoss()

    best_model_path = os.path.join(save_path, "best-ae-joiner.pth")
    last_model_path = os.path.join(save_path, "last-ae-joiner.pth")

    best_precision_sum = 0.0

    for epoch in range(epochs):
        model.train()

        epoch_losses = []
        epoch_labels = [0.0] * len(train_dataset)
        epoch_predictions = []

        for negative_batch in train_loader:
            negative_batch = negative_batch.to(device)

            epoch_labels.extend([0.0] * negative_batch.size(dim=0))

            _, reconstructed = model(negative_batch)
            loss = criterion(reconstructed, negative_batch)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            epoch_losses.append(loss.cpu().item())
        
        percentile = 0.95
        epoch_losses = sorted(epoch_losses)
        index_percentile = int(percentile * len(epoch_losses))
        threshold = epoch_losses[index_percentile]
        print(f"Threshold for epoch {epoch+1} is {threshold:.4f}")

        num_negative_predictions = int(percentile * len(epoch_labels))
        epoch_predictions.extend([0.0] * num_negative_predictions)
        epoch_predictions.extend([1.0] * (len(epoch_labels) - num_negative_predictions))

        for positive_edge in train_dataset.positive_data:
            positive_edge = positive_edge.to(device)
            epoch_labels.append(1.0)

            with torch.no_grad():
                _, reconstructed = model(positive_edge)
                loss = criterion(reconstructed, positive_edge).cpu().item()

            if loss > threshold:
                epoch_predictions.append(1.0)
            else:
                epoch_predictions.append(0.0)

        print(f"Epoch {epoch+1} finished.")
        print("TRAIN REPORT:")
        print(classification_report(epoch_labels, epoch_predictions, digits=4))

        print("EVALUATION REPORT:")
        _, book_predictions, book_labels = evaluate(model, val_graphs_book, device, criterion, "book", threshold)
        _, dict_predictions, dict_labels = evaluate(model, val_graphs_dict, device, criterion, "dictionary", threshold)
        _, peri_predictions, peri_labels = evaluate(model, val_graphs_peri, device, criterion, "periodical", threshold)
        print()

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
        
    torch.save(
        {**model.dict_for_saving, "classification_threshold": threshold},
        last_model_path,
    )


def main():
    args = parse_arguments()
    logging.basicConfig(
        level=logging.INFO,
        force=True,
    )
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

    # Save normalizer
    logging.info("Saving normalizer ...")
    with open(os.path.join(args.save, "normalizer.pkl"), 'wb') as f:
        pickle.dump(normalizer, f)
    logging.info("Normalizer created and saved.")

    logging.info("Flattening train graphs ...")
    train_dataset = Dataset(train_graphs)
    logging.info("Train graphs flattened.")

    # Model
    logging.info("Creating model ...")
    input_size = train_graphs[0].node_features.size(dim=1) * 2 + train_graphs[0].edge_attr.size(dim=1)
    model = AutoEncoder(
        input_size=input_size,
        encoding_size=args.encoding_size,
        device=device,
    )
    print(model)
    model = model.to(device)
    logging.info("Model created.")

    logging.info("Starting training ...")
    start = perf_counter()
    train(
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_graphs_book=val_graphs_book,
        val_graphs_dict=val_graphs_dict,
        val_graphs_peri=val_graphs_peri,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save,
    )
    end = perf_counter()
    t = end - start
    logging.info(f"Training finished. Took {t:.1f} s")


if __name__ == "__main__":
    main()
