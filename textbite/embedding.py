import sys
import argparse
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

from torch import FloatTensor
import torch

from safe_gpu import safe_gpu

from textbite.utils import LineLabel, CZERT_PATH
from textbite.language_model import create_language_model
from textbite.geometry import PageGeometry, LineGeometry


@dataclass
class LineEmbedding:
    embedding: FloatTensor
    page_id: str
    line_id: str
    label: Optional[LineLabel] = None
    bite_id: Optional[str] = None


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--line-mapping", default=None, type=str, help="Path to a line id -- line label mapping file.")
    parser.add_argument("--bite-mapping", default=None, type=str, help="Path to a line id -- bite id mapping file.")
    parser.add_argument("--model", default=CZERT_PATH, type=str, help="Path to a custom BERT model.")
    parser.add_argument("--save", default=".", type=str, help="Where to save the result pickle file.")

    args = parser.parse_args()
    return args


class EmbeddingProvider:
    def __init__(self, device, bert_path: str=CZERT_PATH):
        self.tokenizer, self.bert = create_language_model(device, bert_path)
        self.bert = self.bert.to(device)
        self.bert.eval()
        self.device = device

    def get_embedding(
        self,
        line_geometry: LineGeometry,
        right_context: str,
        max_len: int = 256,
    ) -> FloatTensor:
        tokenizer_output = self.tokenizer(
            line_geometry.text_line.transcription.strip(),
            right_context,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenizer_output["input_ids"].to(self.device)
        token_type_ids = tokenizer_output["token_type_ids"].to(self.device)
        attention_mask = tokenizer_output["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.bert(
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

    id_to_label_dict = {}   
    if args.line_mapping:
        with open(args.line_mapping, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

        for mapping_line in lines:
            line_id, label = mapping_line.split("\t")
            line_id = line_id.strip()
            label = LineLabel(int(label.strip()))
            id_to_label_dict[line_id] = label

    id_to_bite_id_dict = {}
    if args.bite_mapping:
        with open(args.bite_mapping, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

        for mapping_line in lines:
            line_id, bite_id = mapping_line.split("\t")
            line_id = line_id.strip()
            bite_id = bite_id.strip()
            id_to_bite_id_dict[line_id] = bite_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_provider = EmbeddingProvider(device, args.model)

    samples = []
    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xml) if xml_filename.endswith(".xml")]
    for xml_filename in xml_filenames:
        xml_path = os.path.join(args.xml, xml_filename)
        geometry = PageGeometry(xml_path)

        for line in geometry.lines:
            label = id_to_label_dict[line.text_line.id] if line.text_line.id in id_to_label_dict else None
            bite_id_ = id_to_bite_id_dict[line.text_line.id] if line.text_line.id in id_to_bite_id_dict else None
            right_context = line.child.text_line.transcription.strip() if line.child else ""

            embedding = embedding_provider.get_embedding(line, right_context=right_context)
            sample = LineEmbedding(embedding, geometry.pagexml.id, line.text_line.id, label=label, bite_id=bite_id_)
            samples.append(sample)

        logging.info(f"Processed {xml_path}")

    save_path = os.path.join(args.save, "graph-embeddings.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(samples, f)


if __name__ == "__main__":
    main()
