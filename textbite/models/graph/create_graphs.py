import sys
import argparse
import logging
import os
import pickle

import torch

from textbite.geometry import PageGeometry
from textbite.embedding import LineEmbedding


class Graph:
    def __init__(self):
        self.node_features = [] # Shape (n_nodes, n_features)
        self.edge_index = [] # Shape (2, n_edges)
        self.edge_attr = [] # Shape (n_edges, n_features)
        self.labels = [] # Shape (1, n_edges)

    def to_pt(self):
        self.node_features = torch.stack(self.node_features)
        self.edge_index = torch.tensor(self.edge_index)
        self.edge_attr = torch.tensor(self.edge_attr)
        self.labels = torch.tensor(self.labels)

    def __str__ (self):
        result_str = ""

        result_str += f"Node features shape: {self.node_features.shape}\n"
        result_str += f"Edge index shape:    {self.edge_index.shape}\n"
        result_str += f"Edge attr shape:     {self.edge_attr.shape}\n"
        result_str += f"Labels ({len(self.labels)}): {self.labels}\n"

        return result_str


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xmls.")
    parser.add_argument("--embeddings", required=True, type=str, help="Path to a pickle file with LineGeometry embeddings")
    parser.add_argument("--save", default=".", type=str, help="Where to save the result pickle file.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    with open(args.embeddings, "rb") as f:
        node_embeddings = pickle.load(f)
        node_embeddings_dict = {embedding.line_id: embedding for embedding in node_embeddings}

    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xml) if xml_filename.endswith(".xml")]
    graphs = []

    for xml_filename in xml_filenames:
        xml_path = os.path.join(args.xml, xml_filename)

        geometry = PageGeometry(path=xml_path)
        geometry.set_neighbourhoods()
        geometry.set_visibility()

        g = Graph()
        from_indices = []
        to_indices = []
        for line_idx, line in enumerate(geometry.lines):
            embedding = node_embeddings_dict[line.text_line.id].embedding
            g.node_features.append(embedding)

            connected_lines = line.neighbourhood + line.visible_lines
            for connected_line in connected_lines:
                to_idx = geometry.lines.index(connected_line)
                from_indices.append(line_idx)
                to_indices.append(to_idx)
                
                line_bite_id = node_embeddings_dict[line.text_line.id].bite_id
                connected_line_bite_id = node_embeddings_dict[connected_line.text_line.id].bite_id
                g.labels.append(int(line_bite_id == connected_line_bite_id))

        g.edge_index.extend([from_indices, to_indices])

        g.to_pt()
        graphs.append(g)
        logging.info(f"Processed: {xml_path}")

    save_path = os.path.join(args.save, "graphs.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)


if __name__ == "__main__":
    main()
