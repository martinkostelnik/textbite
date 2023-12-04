import sys
import os
import argparse
import logging
from typing import Tuple
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from safe_gpu import safe_gpu

from textbite.models.baseline.dataset import BaselineDataset
from textbite.models.baseline.model import BaselineModel


def confusion_report(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    return f'{" ".join(cn for cn in class_names)}\n{cm}'


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a pickle file with training data.")

    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("-r", "--train-ratio", type=float, default=0.8, help="Train/Validation ratio.")

    parser.add_argument("-l", "--nb-hidden", type=int, default=2, help="Number of layers in the model.")
    parser.add_argument("-n", "--hidden-width", type=int, default=256, help="Hidden size of layers.")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help="Dropout probability in the model.")

    parser.add_argument("-e", "--max-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument("--save", type=str, help="Where to save the model best by validation F1")
    parser.add_argument("--checkpoint-dir", type=str, help="Where to put all training checkpoints")

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
        save_path: str,
        checkpoint_dir: str,
    ):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.4, 0.4]).to(device))
    # criterion = torch.nn.CrossEntropyLoss()

    best_f1_val = 0.0

    for epoch in range(epochs):
        epoch_labels = []
        epoch_preds = []

        epoch_labels_val = []
        epoch_preds_val = []

        model.train()
        train_nb_examples = 0
        train_loss = 0.0
        for step, (embeddings, labels) in enumerate(train_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            loss.backward()
            optim.step()

            train_loss += loss.cpu().item()
            train_nb_examples += len(labels)

            epoch_labels.extend(labels.cpu())
            epoch_preds.extend(preds.cpu())

        model.eval()
        val_nb_examples = 0
        val_loss = 0.0
        for embeddings, labels in val_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(embeddings)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            val_nb_examples += len(labels)
            val_loss += loss.cpu().item()

            epoch_labels_val.extend(labels.cpu())
            epoch_preds_val.extend(preds.cpu())

        f1 = f1_score(epoch_labels, epoch_preds, average="macro")
        f1_val = f1_score(epoch_labels_val, epoch_preds_val, average="macro")

        dict_for_saving = {
            "state_dict": model.state_dict(),
            "n_layers": model.n_layers,
            "hidden_size": model.hidden_size
        }
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            print(f"SAVING MODEL at f1_val = {f1_val}")
            torch.save(dict_for_saving, os.path.join(save_path, "BaselineModel.pth"))

        print(f"Epoch {epoch + 1} finished | train f1 = {f1:.2f} loss = {train_loss/train_nb_examples:.3e} | val f1 = {f1_val:.2f} loss = {val_loss/val_nb_examples:.3e} {'| Only zeros in last val batch!' if not any(preds) else ''}")
        if (epoch + 1) % 10 == 0:
            target_names = ["None", "Terminating", "Title"]  # should be provided by model or someone like that
            print("TRAIN REPORT:")
            print(classification_report(epoch_labels, epoch_preds, digits=4, zero_division=0, target_names=target_names))
            print(confusion_report(epoch_labels, epoch_preds, target_names))
            print("VALIDATION REPORT:")
            print(classification_report(epoch_labels_val, epoch_preds_val, digits=4, zero_division=0, target_names=target_names))
            print(confusion_report(epoch_labels_val, epoch_preds_val, target_names))
            print("VALIDATION REPORT:")
            if checkpoint_dir:
                torch.save(dict_for_saving, os.path.join(checkpoint_dir, f'checkpoint.{epoch}.pth'))


def main(args):
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on: {device}")

    # Data loaders
    logging.info("Creating data loaders ...")
    train_loader, val_loader = prepare_loaders(args.data, args.batch_size, args.train_ratio)
    logging.info("Data loaders ready.")

    # Model
    logging.info("Creating model ...")
    model = BaselineModel(
        n_layers=args.nb_hidden,
        hidden_size=args.hidden_width,
        dropout_prob=args.dropout,
        device=device,
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
        epochs=args.max_epochs,
        lr=args.lr,
        save_path=args.save,
        checkpoint_dir=args.checkpoint_dir,
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
