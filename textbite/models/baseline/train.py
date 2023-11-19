import sys
import argparse
import logging
from typing import Tuple
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from safe_gpu import safe_gpu

from textbite.models.baseline.dataset import BaselineDataset
from textbite.models.baseline.model import BaselineModel


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a pickle file with training data.")

    parser.add_argument("-b", type=int, default=64, help="Batch size")
    parser.add_argument("-r", type=float, default=0.8, help="Train/Validation ratio.")

    parser.add_argument("-l", type=int, default=2, help="Number of layers in the model.")
    parser.add_argument("-n", type=int, default=256, help="Hidden size of layers.")
    parser.add_argument("-d", type=float, default=0.1, help="Dropout probability in the model.")

    parser.add_argument("-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    return args


def prepare_loaders(path: str, batch_size: int, ratio: float) -> Tuple[DataLoader, DataLoader]:
    start = perf_counter()
    dataset = BaselineDataset(path)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [ratio, 1 - ratio])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    end = perf_counter()
    t = end - start
    logging.info(f"Train data loaded. n_samples = {len(dataset)}\ttrain = {len(train_dataset)}\tval = {len(val_dataset)}\ttook {t:.1f} s")

    return train_loader, val_loader


def train(
        model: BaselineModel,
        device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
    ):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr)
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.02, 0.49, 0.49]).to(device))
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_labels = []
        epoch_preds = []

        epoch_labels_val = []
        epoch_preds_val = []

        model.train()
        for step, (embeddings, labels) in enumerate(train_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optim.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            # print(loss.item())

            loss.backward()
            optim.step()

            epoch_labels.extend(labels.cpu())
            epoch_preds.extend(preds.cpu())

        model.eval()
        for embeddings, labels in val_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(embeddings)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            epoch_labels_val.extend(labels.cpu())
            epoch_preds_val.extend(preds.cpu())

        f1 = f1_score(epoch_labels, epoch_preds, average="weighted")
        print(f"Epoch {epoch + 1} finished | f1 = {f1:.2f} | Predicts zeros: {not any(preds)}")
        if not (epoch + 1) % 10:
            print("TRAIN REPORT:")
            print(classification_report(epoch_labels, epoch_preds, digits=4, zero_division=0, target_names=["None", "Terminating", "Title"]))
            print("VALIDATION REPORT:")
            print(classification_report(epoch_labels_val, epoch_preds_val, digits=4, zero_division=0, target_names=["None", "Terminating", "Title"]))


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on: {device}")

    # Data loaders
    logging.info("Creating data loaders ...")
    train_loader, val_loader = prepare_loaders(args.data, args.b, args.r)
    logging.info("Data loaders ready.")

    # Model
    logging.info("Creating model ...")
    model = BaselineModel(
        n_layers=args.l,
        hidden_size=args.n,
        dropout_prob=args.d,
        device=device,
        context=False,
    )
    model = model.to(device)
    logging.info("Model created.")

    logging.info("Starting training ...")
    start = perf_counter()
    train(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.e,
        lr=args.lr,
    )
    end = perf_counter()
    t = end - start
    logging.info(f"Training finished. Took {t:.1f} s")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        force=True,
    )
    args = parse_arguments()
    safe_gpu.claim_gpus()
    main(args)
