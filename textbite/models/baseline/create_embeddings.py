import argparse
import sys
import pickle
from typing import Optional
from tqdm import tqdm

import torch
from torch import FloatTensor
from transformers import BertModel, BertTokenizerFast

from safe_gpu import safe_gpu

from textbite.models.baseline.utils import Sample
from textbite.utils import LineLabel
from textbite.utils import CZERT_PATH

from semant.language_modelling.model import build_model
from semant.language_modelling.tokenizer import build_tokenizer


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", required=True, type=str, help="Path to a mapping file.")
    parser.add_argument("--model", type=str, help="Path to a custom BERT model.")

    args = parser.parse_args()
    return args


def get_embedding(
        text: str,
        bert: BertModel,
        tokenizer: BertTokenizerFast,
        device,
        max_len: int = 256,
        left_context: Optional[str] = None,
        right_context: Optional[str] = None,
    ) -> FloatTensor:
    # tokenizer_output = tokenizer(
    #     text,
    #     right_context,
    #     max_length=max_len,
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt",
    # )
    tokenizer_output = tokenizer(text, right_context)

    input_ids = tokenizer_output["input_ids"].to(device)
    token_type_ids=tokenizer_output["token_type_ids"].to(device)
    attention_mask=tokenizer_output["attention_mask"].to(device)

    outputs = bert(
        input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )
    outputs = outputs.pooler_output.detach().flatten().cpu()

    return outputs


def main(args):
    with open(args.i, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if line and line != "\n"]

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

    samples = []
    for i, line in tqdm(enumerate(lines)):
        line = line.strip()
        if not line:
            continue

        try:
            right_context, _ = lines[i + 1].split("\t")
        except IndexError:
            right_context = ""

        try:
            left_context, _ = lines[i - 1].split("\t")
        except IndexError:
            left_context = ""

        text, label = line.split("\t")

        embedding = get_embedding(text, bert, tokenizer, device, right_context=right_context)
        sample = Sample(embedding, LineLabel(int(label)))
        samples.append(sample)

    with open("data-516.pkl", "wb") as f:
        pickle.dump(samples, f)


if __name__ == "__main__":
    safe_gpu.claim_gpus()
    args = parse_arguments()
    main(args)
