import logging
import argparse
import sys
import os
import json
from typing import List

import torch
from torch import FloatTensor

from safe_gpu import safe_gpu

from transformers import BertModel, BertTokenizerFast

from pero_ocr.document_ocr.layout import PageLayout

from textbite.utils import CZERT_PATH, LineLabel
from textbite.models.baseline.model import BaselineModel


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a folder with data, either xml or txt.")
    parser.add_argument("--model", required=True, type=str, help="Path to the baseline model.")

    args = parser.parse_args()
    return args


class EmbeddingProvider:
    def __init__(self, czert_path, device):
        self.tokenizer = BertTokenizerFast.from_pretrained(czert_path)
        self.czert = BertModel.from_pretrained(czert_path)
        self.czert.to(device)

        self.device = device

    def get_embedding(
        self,
        text: str,
        right_context: str,
    ) -> FloatTensor:
        tokenizer_output = self.tokenizer(
            text,
            right_context,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenizer_output["input_ids"].to(self.device)
        token_type_ids = tokenizer_output["token_type_ids"].to(self.device)
        attention_mask = tokenizer_output["attention_mask"].to(self.device)

        with torch.no_grad():
            czert_outputs = self.czert(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        return czert_outputs.pooler_output


class LineClassifier:
    def __init__(self, model_path, device):
        model_checkpoint = torch.load(model_path, map_location=device)
        self.model = BaselineModel(
            device=device,
            n_layers=model_checkpoint["n_layers"],
            hidden_size=model_checkpoint["hidden_size"]
        )
        self.model.to(device)
        self.device = device

    def classify_line(self, features):
        with torch.no_grad():
            model_outputs = self.model(features).cpu()
        prediction = torch.argmax(model_outputs)
        return LineLabel(prediction.item())


def infer_pagexml(
    pagexml: PageLayout,
    embedding_provider,
    line_classifier,
) -> List[List[str]]:
    lines = [line for region in pagexml.regions for line in region.lines]
    result = []
    bite = []
    for i, line in enumerate(lines):
        if line.transcription is None:
            logging.warning(f'Line {line.id} has no transcription')
            continue

        text = line.transcription.strip()
        try:
            right_context = lines[i + 1].transcription.strip()
        except IndexError:
            right_context = ""

        embedding = embedding_provider.get_embedding(text, right_context)
        predicted_class = line_classifier.classify_line(embedding)

        bite.append(line.id)
        if predicted_class == LineLabel.TERMINATING:
            result.append(bite)
            bite = []

    if bite:
        result.append(bite)

    return result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_provider = EmbeddingProvider(CZERT_PATH, device)
    line_classifier = LineClassifier(args.model, device)

    for filename in os.listdir(args.data):
        if not filename.endswith(".xml"):
            continue

        path = os.path.join(args.data, filename)
        logging.debug(f"Processing: {path}")
        pagexml = PageLayout(file=path)

        result = infer_pagexml(pagexml, embedding_provider, line_classifier)

        out_path = os.path.join(args.data, filename.replace(".xml", ".json"))
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        force=True,
    )
    args = parse_arguments()
    safe_gpu.claim_gpus()
    main(args)
