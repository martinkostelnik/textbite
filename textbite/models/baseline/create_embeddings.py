import argparse
import sys
import pickle
from typing import Optional
import logging
import os

import torch
from torch import FloatTensor
from transformers import BertModel, BertTokenizerFast

from safe_gpu import safe_gpu

from textbite.models.baseline.utils import Sample
from textbite.geometry import PageGeometry, LineGeometry
from textbite.utils import LineLabel
from textbite.utils import CZERT_PATH

from semant.language_modelling.model import build_model
from semant.language_modelling.tokenizer import build_tokenizer


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--mapping", required=True, type=str, help="Path to a mapping file.")
    parser.add_argument("--model", type=str, help="Path to a custom BERT model.")
    parser.add_argument("--save", default=".", type=str, help="Where to save the result pickle file.")

    args = parser.parse_args()
    return args


def get_embedding(
        line_geometry: LineGeometry,
        bert: BertModel,
        tokenizer: BertTokenizerFast,
        device,
        right_context: str,
        max_len: int = 256,
    ) -> FloatTensor:
    tokenizer_output = tokenizer(
        line_geometry.text_line.transcription.strip(),
        right_context,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # tokenizer_output = tokenizer(line_geometry.text_line.transcription.strip(), right_context)1

    input_ids = tokenizer_output["input_ids"].to(device)
    token_type_ids=tokenizer_output["token_type_ids"].to(device)
    attention_mask=tokenizer_output["attention_mask"].to(device)

    outputs = bert(
        input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )
    bert_embedding = outputs.pooler_output.detach().flatten().cpu()
    geometry_embedding = line_geometry.get_features(return_type="pt")

    outputs = torch.cat([bert_embedding, geometry_embedding])

    return outputs


def main():
    args = parse_arguments()
    safe_gpu.claim_gpus()
    logging.basicConfig(level=args.logging_level, force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model:
        checkpoint = torch.load(args.model)
        tokenizer = build_tokenizer(
            seq_len=checkpoint["seq_len"],
            fixed_sep=checkpoint["fixed_sep"],
            masking_prob=0.0,
        )

        bert = build_model(
            czert=checkpoint["czert"],
            vocab_size=len(tokenizer),
            device=device,
            seq_len=checkpoint["seq_len"],
            out_features=checkpoint["features"],
            mlm_level=0,
            sep=checkpoint["sep"],
        )
        bert.bert.load_state_dict(checkpoint["bert_state_dict"])
        bert.nsp_head.load_state_dict(checkpoint["nsp_head_state_dict"])
    else:
        tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
        bert = BertModel.from_pretrained(CZERT_PATH)

    bert = bert.to(device)

    with open(args.mapping, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    lines_dict = {}

    for mapping_line in lines:
        line_id, label = mapping_line.split("\t")
        line_id = line_id.strip()
        label = LineLabel(int(label.strip()))
        lines_dict[line_id] = label

    samples = []
    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xml) if xml_filename.endswith(".xml")]
    for xml_filename in xml_filenames:
        xml_path = os.path.join(args.xml, xml_filename)
        geometry = PageGeometry(xml_path)

        for line in geometry.lines:
            try:
                label = lines_dict[line.text_line.id]
            except KeyError:
                continue
            right_context = line.child.text_line.transcription.strip() if line.child else ""

            embedding = get_embedding(line, bert, tokenizer, device, right_context=right_context)
            sample = Sample(embedding, label)
            samples.append(sample)

    save_path = os.path.join(args.save, "data-combined-lm72.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(samples, f)


if __name__ == "__main__":
    main()
