import argparse
from pathlib import Path

import hdbscan
import torch
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from textbite.mlp_contrastive.dataset import TextBitePageDataset
from textbite.mlp_contrastive.train import TextBiteMLPEncoder
from textbite.visualization.utils import COLORS

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--coco", type=Path, default=None)
    parser.add_argument("--img", type=Path, required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ALPHA = 0.4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output.mkdir(parents=True, exist_ok=True)

    dataset = TextBitePageDataset(
        text_embedding_dir=args.embeddings,
        xml_dir=args.xml,
        coco_path=args.coco,
    )

    checkpoint = torch.load(args.model)
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model = TextBiteMLPEncoder(
        hidden_dim=checkpoint["hyper_parameters"]["hidden_dim"],
        text_embedder_input_dim=checkpoint["hyper_parameters"]["text_embedder_input_dim"],
        n_layers=checkpoint["hyper_parameters"]["n_layers"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        metric="precomputed",
    )

    import sys
    np.set_printoptions(threshold=sys.maxsize, suppress=True)
    for sample in dataset:
        file_name = sample["file_name"]
        bboxes = torch.tensor(sample["bbox"]).to(device)
        text_embeddings = torch.tensor(sample["text_embedding"]).to(device)

        embedding = model(
            bboxes=bboxes.unsqueeze(0),
            text_embeddings=text_embeddings.unsqueeze(0),
        ).squeeze(0)
        
        embedding = embedding.detach().cpu().numpy()
        distances = cosine_distances(embedding).astype(np.float64)
        labels = clusterer.fit_predict(distances)

        max_label = labels.max()
        for i in range(len(labels)):
            if labels[i] == -1:
                max_label += 1
                labels[i] = max_label

        img_path = args.img / f"{file_name}.jpg"
        img = cv2.imread(str(img_path))
        overlay = img.copy()
        for bbox, label in zip(sample["bbox"], labels):
            cx, cy, w, h = map(float, bbox)
            color = COLORS[label % len(COLORS)]

            xmin = int((cx - w / 2) * img.shape[1])
            ymin = int((cy - h / 2) * img.shape[0])
            xmax = int((cx + w / 2) * img.shape[1])
            ymax = int((cy + h / 2) * img.shape[0])

            # Draw filled rectangle with some transparency
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, -1)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 4)

        cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0, img)
        
        out_path = args.output / f"{file_name}.jpg"
        cv2.imwrite(str(out_path), img)

if __name__ == "__main__":
    main()
