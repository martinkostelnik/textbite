import logging
import argparse
import sys
import os
import json
from typing import List

import numpy as np
import torch

import cv2
from textbite.visualization.utils import overlay_line

from safe_gpu import safe_gpu

from pero_ocr.document_ocr.layout import PageLayout

from textbite.utils import CZERT_PATH, LineLabel
from textbite.models.baseline.model import BaselineModel
from textbite.embedding import EmbeddingProvider
from textbite.geometry import PageGeometry


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--data", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--images", type=str, help="Path to a folder with images data. Used for extra visualizations.")
    parser.add_argument("--diagnostics", type=str, help="Path to a folder where to place diagnostic images")
    parser.add_argument("--lm", default=CZERT_PATH, type=str, help="Path to language model used for text embedding.")
    parser.add_argument("--model", required=True, type=str, help="Path to the baseline model.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")

    args = parser.parse_args()
    return args


class LineClassifier:
    def __init__(self, model_path: str, device):
        model_checkpoint = torch.load(model_path, map_location=device)
        self.model = BaselineModel(
            device=device,
            n_layers=model_checkpoint["n_layers"],
            hidden_size=model_checkpoint["hidden_size"]
        )
        self.model.load_state_dict(model_checkpoint["state_dict"])
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def __call__(self, features):
        features = features.to(self.device)
        with torch.no_grad():
            model_outputs = self.model(features).cpu()
        prediction = torch.argmax(model_outputs)
        return LineLabel(prediction.item())

    def probs(self, features):
        features = features.to(self.device)
        with torch.no_grad():
            model_outputs = self.model(features).cpu()
        return torch.softmax(model_outputs, -1).numpy()


def get_geometry_probs(
    pagexml: PageLayout,
    embedding_provider: EmbeddingProvider,
    line_classifier: LineClassifier,
) -> PageGeometry:
    page_geometry = PageGeometry(pagexml=pagexml)

    for line in page_geometry.lines:
        if not line.text_line.transcription.strip():
            logging.warning(f'Line {line.text_line.id} has no transcription')
            continue

        right_context = line.child.text_line.transcription.strip() if line.child else ""

        embedding = embedding_provider.get_embedding(line, right_context)
        line.textbite_baseline_probs = line_classifier.probs(embedding)  # TODO would be cool to have a clean way

    return page_geometry


def infer_pagexml(page_geometry: PageGeometry) -> List[List[str]]:
    result = []
    bite = []
    for line in page_geometry.lines:
        if not line.text_line.transcription.strip():
            logging.warning(f'Line {line.text_line.id} has no transcription')
            continue

        predicted_class = np.argmax(line.textbite_baseline_probs)

        bite.append(line.text_line.id)
        if predicted_class == LineLabel.TERMINATING:
            result.append(bite)
            bite = []

    if bite:
        result.append(bite)

    return result


def draw_probs(img, page_geometry):
    for line in page_geometry.lines:
        if not line.text_line.transcription.strip():
            logging.warning(f'Line {line.text_line.id} has no transcription')
            continue

        img = overlay_line(img, line, tuple(255*x for x in line.textbite_baseline_probs), alpha=0.3)

    return img


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    safe_gpu.claim_gpus()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_provider = EmbeddingProvider(device, args.lm)
    line_classifier = LineClassifier(args.model, device)

    os.makedirs(args.save, exist_ok=True)
    xml_filenames = [xml_filename for xml_filename in os.listdir(args.data) if xml_filename.endswith(".xml")]

    for filename in xml_filenames:
        path = os.path.join(args.data, filename)
        logging.info(f"Processing: {path}")
        pagexml = PageLayout(file=path)

        page_geometry = get_geometry_probs(pagexml, embedding_provider, line_classifier)
        result = infer_pagexml(page_geometry)

        out_path = os.path.join(args.save, filename.replace(".xml", ".json"))
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)

        if args.diagnostics:
            os.makedirs(args.diagnostics, exist_ok=True)

            path_img = os.path.join(args.images, filename.replace(".xml", ".jpg"))
            img = cv2.imread(path_img)

            if img is None:
                logging.warning(f'Failed to load {path_img}, skipping diagnostics')
                continue

            img = draw_probs(img, page_geometry)

            res_filename = os.path.join(args.diagnostics, filename.replace(".xml", "-diag.jpg"))
            cv2.imwrite(res_filename, img)


if __name__ == "__main__":
    main()
