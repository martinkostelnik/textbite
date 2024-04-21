import argparse
import logging
import os
from typing import List
from dataclasses import dataclass

from safe_gpu import safe_gpu
from transformers import BertModel, BertTokenizerFast
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from pero_ocr.document_ocr.layout import PageLayout

from textbite.bite import load_bites, save_bites, Bite
from textbite.utils import CZERT_PATH
from textbite.models.joiner.infer import join_bites_by_dict


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--data", required=True, type=str, help="Path to a folder with jsons containing bites.")
    parser.add_argument("--xmls", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")

    return parser.parse_args()


@dataclass
class BiteEmbedding:
    pooler_output: np.ndarray
    cls_output: np.ndarray


def czert_forward(text, tokenizer, czert, device):
    tokenized_text = tokenizer(
        text,
        max_length=512,
        return_tensors="pt",
        truncation=True,
    )

    input_ids = tokenized_text["input_ids"].to(device)
    token_type_ids = tokenized_text["token_type_ids"].to(device)
    attention_mask = tokenized_text["attention_mask"].to(device)

    with torch.no_grad():
        czert_outputs = czert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    pooler_output = czert_outputs.pooler_output.cpu().numpy()
    cls_output = czert_outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return pooler_output, cls_output


def get_bite_embedding(bite: Bite, pagexml: PageLayout, tokenizer, czert, device) -> BiteEmbedding:
    text_lines = [text_line for text_line in pagexml.lines_iterator() if text_line.id in bite.lines]
    bite_transcription = " ".join([line.transcription.strip() for line in text_lines])
    bite_embedding = BiteEmbedding(*czert_forward(bite_transcription, tokenizer, czert, device))

    return bite_embedding


def cluster_bites(
        bites: List[Bite],
        pagexml: PageLayout,
        tokenizer,
        czert,
        device,
    ) -> List[Bite]:
    bite_pooler_embeddings = np.ndarray(shape=(len(bites), 768))
    bite_cls_embeddings = np.ndarray(shape=(len(bites), 768))
    for i, bite in enumerate(bites):
        embedding = get_bite_embedding(bite, pagexml, tokenizer, czert, device)
        bite_pooler_embeddings[i, :] = embedding.pooler_output
        bite_cls_embeddings[i, :] = embedding.cls_output

    model_cls = AgglomerativeClustering(n_clusters=None, distance_threshold=7)
    model_cls.fit(bite_cls_embeddings)

    labels = model_cls.labels_

    label_dict = [[] for _ in range(len(labels))]
    for idx, label in enumerate(labels.tolist()):
        label_dict[label].append(idx)

    label_dict = {cluster[0]: cluster[1:] for cluster in label_dict if cluster}
    new_bites = join_bites_by_dict(label_dict, bites)

    return new_bites


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    safe_gpu.claim_gpus()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Creating embeddings on {device}")

    logging.info("Loading tokenizer ...")
    tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
    logging.info("Tokenizer loaded.")
    logging.info("Loading CZERT ...")
    czert = BertModel.from_pretrained(CZERT_PATH)
    czert = czert.to(device)
    logging.info("CZERT loaded.")

    logging.info("Creating save directory ...")
    os.makedirs(args.save, exist_ok=True)
    logging.info("Save directory created.")

    json_filenames = [filename for filename in os.listdir(args.data) if filename.endswith(".json")]

    logging.info("Starting inference ...")
    for i, json_filename in enumerate(json_filenames):
        xml_filename = json_filename.replace(".json", ".xml")
        filename = json_filename.replace(".json", "")

        path_json = os.path.join(args.data, json_filename)
        path_xml = os.path.join(args.xmls, xml_filename)
        save_path = os.path.join(args.save, json_filename)

        logging.info(f"({i}/{len(json_filenames)}) | Processing: {path_json}")

        pagexml = PageLayout(file=path_xml)
        original_bites = load_bites(path_json)

        if len(original_bites) == 1:
            logging.info(f"Single region detected on {xml_filename}, saving as is.")
            new_bites = original_bites
        else:
            new_bites = cluster_bites(
                original_bites,
                pagexml,
                tokenizer,
                czert,
                device,
            )

        save_bites(new_bites, save_path)


if __name__ == '__main__':
    main()
