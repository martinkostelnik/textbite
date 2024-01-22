import argparse
import logging
import os
from typing import Optional
import pickle

from ultralytics import YOLO
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from pero_ocr.document_ocr.layout import PageLayout

from textbite.data_processing.label_studio import LabelStudioExport, AnnotatedDocument, best_intersecting_bbox
from textbite.models.yolo.infer import YoloBiter
from textbite.geometry import AABB, polygon_to_bbox


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--json", required=True, type=str, help="Path to label studio exported json.")
    parser.add_argument("--model", required=True, type=str, help="Path to the .pt file with weights of YOLO model.")
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--xmls", type=str, help="Path to a folder with xml data.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")

    return parser.parse_args()


class Graph:
    def __init__(self, id: str, node_features, from_indices, to_indices, labels):
        self.graph_id = id

        self.node_features = torch.stack(node_features)  # Shape (n_nodes, n_features)
        self.edge_index = torch.tensor([from_indices, to_indices], dtype=torch.int64)  # Shape (2, n_edges)
        self.edge_attr = torch.tensor([])  # Shape (n_edges, n_features), but we have none
        self.labels = torch.tensor(labels)  # Shape (1, n_edges)

    def __str__(self):
        output_str = ""

        output_str += f"Graph id = {self.graph_id}\n"
        output_str += f"Node features shape = {self.node_features.shape}\n"
        output_str += f"Edge index shape = {self.edge_index.shape}\n"
        output_str += f"Edge attr shape = {self.edge_attr.shape}\n"
        output_str += f"Labels shape = {self.labels.shape}\n"
        output_str += f"Positive edges = {torch.sum(self.labels)}"

        return output_str


def get_graph_from_file(model: YoloBiter, path_img: str, pagexml: PageLayout, document: Optional[AnnotatedDocument]=None) -> Graph:
    texts, titles = model.find_bboxes(path_img)
    texts = model.filter_bboxes(texts)
    titles = model.filter_bboxes(titles)
    
    id = os.path.basename(path_img).replace(".jpg", "")

    if document is not None:
        mapping = {}
        for yolo_region in texts + titles:
            idx = best_intersecting_bbox(yolo_region, [r.bbox for r in document.regions])
            mapping[yolo_region] = None if idx is None else document.regions[idx].id

    transcriptions = ["" for _ in texts + titles]
    for line in pagexml.lines_iterator():
        line_bbox = polygon_to_bbox(line.polygon)

        idx = best_intersecting_bbox(line_bbox, texts + titles)
        if idx is not None and line and line.transcription and line.transcription.strip():
            transcriptions[idx] += f"{line.transcription.strip()}\n"

    vectorizer = TfidfVectorizer(max_features=64)
    try:
        tfidf_matrix = vectorizer.fit_transform(transcriptions)
    except ValueError:
        raise RuntimeError("XML Transcriptions corrupted")
    tfidf_vectors = tfidf_matrix.toarray()

    node_features = []
    labels = []
    from_indices = []
    to_indices = []
    for from_idx, yolo_region in enumerate(texts + titles):
        features = tfidf_vectors[from_idx, :]
        if len(features) != 64:
            to_add = 64 - len(features)
            features = features.tolist()
            features.extend([0.0] * to_add)
        node_features.append(torch.tensor(features, dtype=torch.float32))

        if document is not None:
            target = document.relations[mapping[yolo_region]][0] if mapping[yolo_region] in document.relations else None
        
        for to_idx, connected_yolo_region in enumerate(texts + titles):
            if yolo_region is connected_yolo_region:
                continue

            from_indices.append(from_idx)
            to_indices.append(to_idx)
            if document is not None:
                label = 1 if connected_yolo_region in mapping and target == mapping[connected_yolo_region] else 0
                labels.append(label)

    return Graph(id, node_features, from_indices, to_indices, labels)


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    export = LabelStudioExport(args.json)
    yolo = YoloBiter(YOLO(args.model))

    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xmls) if xml_filename.endswith(".xml")]
    bad_files = 0
    graphs = []

    for filename in xml_filenames:
        path_xml = os.path.join(args.xmls, filename)
        path_img = os.path.join(args.images, filename.replace(".xml", ".jpg"))
        
        logging.info(f"Processing: {path_xml}")
        try:
            document = export.documents_dict[filename.replace(".xml", ".jpg")]
        except KeyError:
            logging.warning(f"{path_img} not labeled, skipping.")
            continue

        pagexml = PageLayout(file=path_xml)
        document.map_to_pagexml(pagexml)
        try:
            graph = get_graph_from_file(yolo, path_img, pagexml, document)
        except RuntimeError:
            bad_files += 1
            logging.warning(f"Runtime error detected, skipping. (total {bad_files} bad files)")
            continue

        graphs.append(graph)

    save_path = os.path.join(args.save, "graphs.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)


if __name__ == "__main__":
    main()
