import argparse
import sys
import pickle

from torch import FloatTensor
from transformers import BertModel, BertTokenizerFast

from textbite.models.baseline.utils import Sample
from textbite.utils import LineLabel


CZERT_PATH = r"UWB-AIR/Czert-B-base-cased"


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
        max_len: int = 64
    ) -> FloatTensor:
    tokenizer_output = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    outputs = czert(
        tokenizer_output["input_ids"],
        token_type_ids=tokenizer_output["token_type_ids"],
        attention_mask=tokenizer_output["attention_mask"]
    )
    outputs = outputs.pooler_output.detach().flatten()

    return outputs


def main(args):
    with open(args.i, "r") as f:
        lines = f.readlines()

    tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)   
    czert = BertModel.from_pretrained(CZERT_PATH)

    samples = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        text, label = line.split("\t")

        embedding = get_embedding(text, czert, tokenizer)
        sample = Sample(embedding, LineLabel(int(label)))
        samples.append(sample)

    with open("test.pkl", "wb") as f:
        pickle.dump(samples, f)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
