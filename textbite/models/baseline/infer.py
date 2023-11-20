import logging
import argparse
import sys
from typing import List, Dict

import torch
from torch import FloatTensor

from safe_gpu import safe_gpu

from transformers import BertModel, BertTokenizerFast

from textbite.utils import CZERT_PATH, LineLabel
from textbite.models.baseline.model import BaselineModel


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a file containing lines for textbiting.")
    parser.add_argument("--model", required=True, type=str, help="Path to the baseline model.")

    args = parser.parse_args()
    return args


def infer(
    data: List[str],
    model_path: str,
    )-> List[List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
    czert = BertModel.from_pretrained(CZERT_PATH)
    czert = czert.to(device)

    model_checkpoint = torch.load(model_path)
    model = BaselineModel(
        device=device,
        n_layers=model_checkpoint["n_layers"],
        hidden_size=model_checkpoint["hidden_size"]
    )
    model = model.to(device)

    result = []
    bite = []
    for i, line in enumerate(data):
        line = line.strip()

        tokenizer_output = tokenizer(
            line,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = tokenizer_output["input_ids"].to(device)
        token_type_ids = tokenizer_output["token_type_ids"].to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)

        czert_outputs = czert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        embedding = czert_outputs.pooler_output

        model_outputs = model(embedding).cpu()
        prediction = torch.argmax(model_outputs)
        predicted_class = LineLabel(prediction.item())

        bite.append(line)
        if predicted_class == LineLabel.TERMINATING:
            result.append(bite)
            bite = []

    if bite:
        result.append(bite)

    return result


def main(args):
    with open(args.data) as f:
        lines = f.readlines()

    result = infer(lines, args.model)
    result_str = "\n\n".join(["\n".join(r) for r in result])
    print(result_str)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        force=True,
    )
    args = parse_arguments()
    safe_gpu.claim_gpus()
    main(args)
