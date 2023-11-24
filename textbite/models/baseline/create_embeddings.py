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


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", required=True, type=str, help="Path to a mapping file.")

    args = parser.parse_args()
    return args


def get_embedding(
        text: str,
        czert: BertModel,
        tokenizer: BertTokenizerFast,
        device,
        max_len: int = 256,
        left_context: Optional[str] = None,
        right_context: Optional[str] = None,
    ) -> FloatTensor:
    tokenizer_output = tokenizer(
        left_context,
        text,
        right_context,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokenizer_output["input_ids"].to(device)
    token_type_ids=tokenizer_output["token_type_ids"].to(device)
    attention_mask=tokenizer_output["attention_mask"].to(device)

    outputs = czert(
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

    tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
    czert = BertModel.from_pretrained(CZERT_PATH)
    print(czert)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # czert = czert.to(device)

    # samples = []
    # for i, line in tqdm(enumerate(lines)):
    #     line = line.strip()
    #     if not line:
    #         continue

    #     try:
    #         right_context, _ = lines[i + 1].split("\t")
    #     except IndexError:
    #         right_context = ""

    #     try:
    #         left_context, _ = lines[i - 1].split("\t")
    #     except IndexError:
    #         left_context = ""

    #     text, label = line.split("\t")

    #     embedding = get_embedding(text, czert, tokenizer, device, left_context=left_context, right_context=right_context)
    #     sample = Sample(embedding, LineLabel(int(label)))
    #     samples.append(sample)

    # with open("novy-lrcontext.pkl", "wb") as f:
    #     pickle.dump(samples, f)


if __name__ == "__main__":
    safe_gpu.claim_gpus()
    args = parse_arguments()
    main(args)
