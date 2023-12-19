import sys
import argparse
import copy
import logging
import os
import pickle
import json
from time import perf_counter

from safe_gpu import safe_gpu

import torch
from torch_geometric.utils import is_undirected

from textbite.geometry import PageGeometry
from textbite.embedding import LineEmbedding
from textbite.language_model import create_language_model


class Graph:
    def __init__(self, graph_id, geometry, node_embeddings_dict, tokenizer, bert, device, yolos_path):
        self.graph_id = graph_id
        node_features = []
        edge_attr = []
        labels = []
        line_ids = []

        from_indices = []
        to_indices = []
        labels = []
        with open(os.path.join(yolos_path, f"{geometry.pagexml.id}.json")) as f:
            bites = json.load(f)

        for line_idx, line in enumerate(geometry.lines):
            embedding = node_embeddings_dict[line.text_line.id].embedding[-25:]
            node_features.append(embedding)
            line_ids.append(line.text_line.id)

            line_bite_id = node_embeddings_dict[line.text_line.id].bite_id

            bert_embedding = node_embeddings_dict[line.text_line.id].embedding[:72]

            connected_lines = list(set(line.neighbourhood + line.visible_lines))
            for connected_line in connected_lines:
                to_idx = geometry.lines.index(connected_line)
                connected_line_bite_id = node_embeddings_dict[connected_line.text_line.id].bite_id
                label = int(line_bite_id == connected_line_bite_id)

                from_indices.append(line_idx)
                to_indices.append(to_idx)
                labels.append(label)

                nsp_prob = get_nsp_prob(tokenizer, bert, device, line.text_line.transcription.strip(), connected_line.text_line.transcription.strip())
                bert_embedding_connected = node_embeddings_dict[connected_line.text_line.id].embedding[:72]

                dist = (bert_embedding - bert_embedding_connected).pow(2).sum().sqrt().item()

                same_bite_yolo = 0.0
                for bite in bites:
                    if line.text_line.id and connected_line.text_line.id in bite:
                        same_bite_yolo = 1.0

                edge_attr.append([same_bite_yolo, dist, nsp_prob])


        # Make graph undirected
        new_from_indices = []
        new_to_indices = []
        new_labels = []
        new_edge_attrs = []
        for from_index, to_index, label, edge_attr_ in zip(from_indices, to_indices, labels, edge_attr):
            reverse_exists = False
            for from_index_, to_index_ in zip(from_indices, to_indices):
                if (from_index, to_index) == (to_index_, from_index_):
                    reverse_exists = True
                    break

            # for from_index_, to_index_ in zip(new_from_indices, new_to_indices):
            #     if (from_index, to_index) == (to_index_, from_index_):
            #         reverse_exists = True
            #         break

            if not reverse_exists:
                new_from_indices.append(to_index)
                new_to_indices.append(from_index)
                new_labels.append(label)
                new_edge_attrs.append(edge_attr_)

        from_indices.extend(new_from_indices)
        to_indices.extend(new_to_indices)
        labels.extend(new_labels)
        edge_attr.extend(new_edge_attrs)

        self.node_features = torch.stack(node_features)  # Shape (n_nodes, n_features)
        self.line_ids = line_ids  # List of strings, (n_nodes, )
        self.edge_index = torch.tensor([from_indices, to_indices])  # Shape (2, n_edges)
        self.edge_attr = torch.tensor(edge_attr)  # Shape (n_edges, n_features), but we have none
        self.labels = torch.tensor(labels)  # Shape (1, n_edges)

    def __str__(self):
        result_str = ""

        result_str += f"Graph ID: {self.graph_id}\n"
        result_str += f"Undirected: {is_undirected(self.edge_index)}\n"
        result_str += f"Node features shape: {self.node_features.shape}\n"
        result_str += f"Edge index shape:    {self.edge_index.shape}\n"
        result_str += f"Edge attr shape:     {self.edge_attr.shape}\n"
        result_str += f"Labels ({len(self.labels)}): {self.labels}\n"

        return result_str


def collate_custom_graphs(graphs):
    multi_graph = copy.deepcopy(graphs[0])

    multi_graph.id = None  # We don't guarantee this makes any sense
    multi_graph.node_features = torch.concatenate([g.node_features for g in graphs])  # Shape (n_nodes, n_features)
    multi_graph.line_ids = None  # We don't guarantee this makes any sense

    adjusted_edges_index = []
    offset = 0
    for g in graphs:
        adjusted_edges_index.append(g.edge_index + offset)
        offset += len(g.node_features)
    multi_graph.edge_index = torch.concatenate(adjusted_edges_index, dim=1)  # Shape (2, n_edges)

    multi_graph.edge_attr = torch.tensor([])  # Shape (n_edges, n_features), but we have none
    multi_graph.labels = torch.concatenate([g.labels for g in graphs])  # Shape (1, n_edges)

    return multi_graph


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--tokenizer" type=str, help="Path to tokenizer.")
    parser.add_argument("--bert", required=True, type=str, help="Path to a bert model")
    parser.add_argument("--yolo", required=True, type=str, help="Path to YOLO model")
    parser.add_argument("--embeddings", required=True, type=str, help="Path to a pickle file with LineGeometry embeddings")
    parser.add_argument("--save", default=".", type=str, help="Where to save the result pickle file.")

    args = parser.parse_args()
    return args


def forward(bert, tokenizer, device, sen1, sen2):
    tokenizer_output = tokenizer(
        sen1.strip(),
        sen2.strip(),
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokenizer_output["input_ids"].to(device)
    token_type_ids = tokenizer_output["token_type_ids"].to(device)
    attention_mask = tokenizer_output["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

    prob = outputs.nsp_output.detach().flatten().cpu()
    return prob


def get_nsp_prob(
        tokenizer,
        bert,
        device,
        sen1: str,
        sen2: str,
    ):
        sen1 = sen1.strip()
        sen2 = sen2.strip()

        prob1 = forward(bert, tokenizer, device, sen1, sen2)
        prob2 = forward(bert, tokenizer, device, sen2, sen1)

        return max(prob1, prob2)


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    safe_gpu.claim_gpus()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, bert = create_language_model(device, args.bert)
    bert = bert.to(device)
    bert.eval()

    with open(args.embeddings, "rb") as f:
        node_embeddings = pickle.load(f)
        node_embeddings_dict = {embedding.line_id: embedding for embedding in node_embeddings}

    all_labeled_filenames = set([embedding.page_id.replace(".jpg", ".xml") for embedding in node_embeddings])
    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xml) if xml_filename.endswith(".xml") and xml_filename in all_labeled_filenames]
    graphs = []

    for xml_filename in xml_filenames:
        t1 = perf_counter()
        xml_path = os.path.join(args.xml, xml_filename)

        geometry = PageGeometry(path=xml_path)
        geometry.set_neighbourhoods()
        geometry.set_visibility()

        graphs.append(Graph(geometry.pagexml.id, geometry, node_embeddings_dict, tokenizer, bert, device, args.yolo))
        t2 = perf_counter()
        logging.info(f"Processed: {xml_path} | Took: {(t2-t1):2f} s")

    save_path = os.path.join(args.save, "graphs.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)


if __name__ == "__main__":
    main()
