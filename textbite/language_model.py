from typing import Tuple

from transformers import BertModel, BertTokenizerFast
import torch

from semant.language_modelling.model import build_model
from semant.language_modelling.tokenizer import build_tokenizer

from textbite.utils import CZERT_PATH


def create_language_model(device, path: str=CZERT_PATH) -> Tuple[BertTokenizerFast, BertModel]:
    if not path or path == CZERT_PATH:
        tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
        bert = BertModel.from_pretrained(CZERT_PATH)
    else:
        checkpoint = torch.load(path)
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

    return tokenizer, bert
