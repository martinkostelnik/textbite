from pathlib import Path
import logging

import lightning as L
from lightning.pytorch.cli import LightningCLI
import torch
from clearml import Task
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from textbite.mlp_contrastive.model import TextBiteMLPEncoder
from textbite.mlp_contrastive.dataset import TextBiteContrastiveDataset


logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, values: list[float]):
        self.val += sum(values)
        self.count += len(values)

    def __call__(self):
        value = self.val / self.count if self.count != 0 else 0.0
        self.reset()
        return value


class TextBiteContrastiveMLP(L.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        text_embedder_input_dim: int,
        lr: float,
        experiment_name: str,
        project_name: str | None = None,
        threshold: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if project_name is not None:
            self.task = Task.init(
                project_name=project_name,
                task_name=experiment_name,
            )

        logger.debug("Initializing TextBiteContrastiveMLP LightningModule ...")

        self.lr = lr
        self.threshold = threshold
        self.experiment_name = experiment_name

        self.model = TextBiteMLPEncoder(
            hidden_dim=hidden_dim,
            text_embedder_input_dim=text_embedder_input_dim,
            n_layers=n_layers,
        )
        print(self.model)

        self.criterion = torch.nn.CosineEmbeddingLoss(reduction="none")

        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        self.train_pos_similarity_meter = AverageMeter()
        self.train_neg_similarity_meter = AverageMeter()
        self.val_pos_similarity_meter = AverageMeter()
        self.val_neg_similarity_meter = AverageMeter()

        self.train_running_labels = []
        self.train_running_predictions = []
        self.val_running_labels = []
        self.val_running_predictions = []

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs["loss"]
        labels = outputs["labels"]
        predictions = outputs["predictions"]
        pos_similarities = outputs["pos_similarities"]
        neg_similarities = outputs["neg_similarities"]

        self.train_loss_meter.update(loss.detach().cpu().numpy().tolist())
        self.train_running_labels.extend(labels)
        self.train_running_predictions.extend(predictions)
        self.train_pos_similarity_meter.update(pos_similarities)
        self.train_neg_similarity_meter.update(neg_similarities)

        return loss.mean()
    
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            accuracy = accuracy_score(self.train_running_labels, self.train_running_predictions)
            f1 = f1_score(self.train_running_labels, self.train_running_predictions)
            precision = precision_score(self.train_running_labels, self.train_running_predictions)
            recall = recall_score(self.train_running_labels, self.train_running_predictions)

            self.log("loss/train", self.train_loss_meter(), prog_bar=True, on_step=True, on_epoch=False)
            self.log("accuracy/train", accuracy, on_step=True, on_epoch=False)
            self.log("f1/train", f1, on_step=True, on_epoch=False)
            self.log("precision/train", precision, on_step=True, on_epoch=False)
            self.log("recall/train", recall, on_step=True, on_epoch=False)
            self.log("similarity/train_pos", self.train_pos_similarity_meter(), on_step=True, on_epoch=False)
            self.log("similarity/train_neg", self.train_neg_similarity_meter(), on_step=True, on_epoch=False)

            self.train_running_labels = []
            self.train_running_predictions = []

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ):
        outputs = self.common_step(batch, batch_idx)
        loss = outputs["loss"]
        labels = outputs["labels"]
        predictions = outputs["predictions"]
        pos_similarities = outputs["pos_similarities"]
        neg_similarities = outputs["neg_similarities"]

        self.val_loss_meter.update(loss.detach().cpu().numpy().tolist())
        self.val_running_labels.extend(labels)
        self.val_running_predictions.extend(predictions)
        self.val_pos_similarity_meter.update(pos_similarities)
        self.val_neg_similarity_meter.update(neg_similarities)

        return loss.mean()
    
    def on_validation_epoch_end(self):
        accuracy = accuracy_score(self.val_running_labels, self.val_running_predictions)
        f1 = f1_score(self.val_running_labels, self.val_running_predictions)
        precision = precision_score(self.val_running_labels, self.val_running_predictions)
        recall = recall_score(self.val_running_labels, self.val_running_predictions)

        self.log("loss/val", self.val_loss_meter(), prog_bar=True)
        self.log("accuracy/val", accuracy)
        self.log("f1/val", f1)
        self.log("precision/val", precision)
        self.log("recall/val", recall)
        self.log("similarity/val_pos", self.val_pos_similarity_meter())
        self.log("similarity/val_neg", self.val_neg_similarity_meter())

        self.val_running_labels = []
        self.val_running_predictions = []

    def common_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        model_output_a, model_output_b = self.forward(batch)
        label = batch["label"]

        loss = self.criterion(
            model_output_a,
            model_output_b, 
            label,
        )

        similarity = torch.nn.functional.cosine_similarity(model_output_a.detach(), model_output_b.detach())
        predictions = (similarity > self.threshold)

        pos_similarities = similarity[label == 1]
        neg_similarities = similarity[label == -1]

        return {
            "loss": loss,
            "labels": (label == 1).detach().cpu().numpy().tolist(),
            "predictions": predictions.detach().cpu().numpy().tolist(),
            "pos_similarities": pos_similarities.detach().cpu().numpy().tolist(),
            "neg_similarities": neg_similarities.detach().cpu().numpy().tolist(),
        }
    
    def forward(
        self,
        batch: dict[torch.Tensor]
    ):
        paragraph_a_bbox = batch["paragraph_a_bbox"]
        paragraph_a_text_embedding = batch["paragraph_a_text_embedding"]
        paragraph_b_bbox = batch["paragraph_b_bbox"]
        paragraph_b_text_embedding = batch["paragraph_b_text_embedding"]

        model_output_a = self.model(
            bboxes=paragraph_a_bbox,
            text_embeddings=paragraph_a_text_embedding,
        )

        model_output_b = self.model(
            bboxes=paragraph_b_bbox,
            text_embeddings=paragraph_b_text_embedding,
        )

        return model_output_a, model_output_b

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class TextBiteContrastiveDataModule(L.LightningDataModule):
    def __init__(
        self,
        text_embedding_dir: Path,
        batch_size: int = 1,
        coco_path_train: Path | None = None,
        coco_path_val: Path | None = None,
        data_limit: int | None = None,
    ):
        super().__init__()

        logger.debug("Initializing TextBiteContrastiveDataModule LightningDataModule ...")

        self.batch_size = batch_size

        self.train_dataset = TextBiteContrastiveDataset(
            coco_path=coco_path_train,
            text_embedding_dir=text_embedding_dir,
            data_limit=data_limit,
        )

        self.val_dataset = TextBiteContrastiveDataset(
            coco_path=coco_path_val,
            text_embedding_dir=text_embedding_dir,
            data_limit=data_limit,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )


def cli_main():
    logging.basicConfig(level=logging.INFO, force=True)

    cli = LightningCLI(
        model_class=TextBiteContrastiveMLP,
        datamodule_class=TextBiteContrastiveDataModule,
    )


if __name__ == "__main__":
    cli_main()
