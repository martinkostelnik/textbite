import logging
import argparse
import sys
import os
import json
from typing import List, Dict

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


def get_embedding(
    text: str,
    right_context: str,
    tokenizer: BertTokenizerFast,
    czert: BertModel,
    device,
    ) -> FloatTensor:
        tokenizer_output = tokenizer(
            text,
            right_context,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = tokenizer_output["input_ids"].to(device)
        token_type_ids = tokenizer_output["token_type_ids"].to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)

        with torch.no_grad():
            czert_outputs = czert(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        return czert_outputs.pooler_output


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
        try:
            right_context = data[i + 1].strip()
        except IndexError:
            right_context = ""

        embedding = get_embedding(line, right_context, tokenizer, czert, device)

        with torch.no_grad():
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


def infer_pagexml(
    data_path: str,
    model_path: str,
    ) -> Dict[str, List[List[str]]]:
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

    results = {}
    for filename in os.listdir(data_path):
        if not filename.endswith(".xml"):
            continue

        path = os.path.join(args.data, filename)
        logging.debug(f"Processing: {path}")
        pagexml = PageLayout(file=path)

        lines = [line for region in pagexml.regions for line in region.lines]
        result = []
        bite = []
        for i, line in enumerate(lines):
            text = line.transcription.strip()
            try:
                right_context = lines[i + 1].transcription.strip()
            except IndexError:
                right_context = ""

            embedding = get_embedding(text, right_context, tokenizer, czert, device)
            with torch.no_grad():
                model_outputs = model(embedding).cpu()
            prediction = torch.argmax(model_outputs)
            predicted_class = LineLabel(prediction.item())

            bite.append(line.id)
            if predicted_class == LineLabel.TERMINATING:
                result.append(bite)
                bite = []

        if bite:
            result.append(bite)

        results[filename] = result
    
    return results

            
def main(args):
    # with open(args.data) as f:
    #     lines = f.readlines()

    # result = infer(lines, args.model)
    # result_str = "\n\n".join(["\n".join(r) for r in result])
    # print(result_str)

    result = infer_pagexml(args.data, args.model)
    for filename, l in result.items():
        pth = os.path.join(args.data, filename.replace(".xml", ".json"))
        with open(pth, "w") as f:
            json.dump(l, f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        force=True,
    )
    args = parse_arguments()
    safe_gpu.claim_gpus()
    main(args)
