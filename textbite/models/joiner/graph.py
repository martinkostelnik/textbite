"""Definition of a graph. Running this file is for debug only.

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


from typing import Optional, List, Dict, Tuple

import itertools
import torch
from transformers import BertModel, BertTokenizerFast

from pero_ocr.core.layout import PageLayout

from textbite.models.yolo.infer import YoloBiter, Bite
from textbite.models.feature_extraction import TextFeaturesProvider, GeometryFeaturesProvider
from textbite.data_processing.label_studio import AnnotatedDocument
from textbite.geometry import best_intersecting_bbox, polygon_to_bbox, AABB, PageGeometry, dist_l2


class Graph:
    def __init__(
        self,
        id: str,
        node_features,
        from_indices,
        to_indices,
        edge_attr,
        labels,
    ):
        self.graph_id = id

        self.node_features = torch.stack(node_features)  # Shape (n_nodes, n_features)
        self.edge_index = torch.tensor([from_indices, to_indices], dtype=torch.int64)  # Shape (2, n_edges)
        self.edge_attr = torch.stack(edge_attr)  # Shape (n_edges, n_features)
        self.labels = torch.tensor(labels)  # Shape (1, n_edges)

    def flatten(self) -> Tuple[List[torch.FloatTensor], List[float], List[Tuple[int, int]]]:
        edge_list = []
        edge_index = self.edge_index.tolist()

        labels = self.labels.tolist()
        labels_to_keep = []

        processed_edges = []

        for i, (from_index, to_index) in enumerate(zip(edge_index[0], edge_index[1])):
            if (to_index, from_index) in processed_edges:
                continue

            labels_to_keep.append(i)
            
            processed_edges.append((from_index, to_index))
            from_features = self.node_features[from_index, :]
            to_features = self.node_features[to_index, :]
            edge_features = self.edge_attr[i, :]

            if from_features[0].item() > to_features[0].item():
                edge = torch.cat([from_features, to_features, edge_features])
            else:
                edge = torch.cat([to_features, from_features, edge_features])

            edge_list.append(edge)

        matched_labels = [labels[idx] for idx in labels_to_keep] if labels else None
        return edge_list, matched_labels, processed_edges

    def __str__(self):
        output_str = ""

        output_str += f"Graph id = {self.graph_id}\n"
        output_str += f"Node features shape = {self.node_features.shape}\n"
        output_str += f"Edge index shape = {self.edge_index.shape}\n"
        output_str += f"Edge attr shape = {self.edge_attr.shape}\n"
        output_str += f"Labels shape = {self.labels.shape}\n"
        output_str += f"Positive edges = {torch.sum(self.labels)}"

        return output_str


class JoinerGraphProvider:
    def __init__(
        self,
        tokenizer: Optional[BertTokenizerFast]=None,
        czert: Optional[BertModel]=None,
        device=None,
        ):
        self.text_features_provider = TextFeaturesProvider(tokenizer, czert, device)
        self.geometric_features_provider = GeometryFeaturesProvider()

    def get_transcriptions(self, regions: List[AABB], pagexml: PageLayout) -> List[str]:
        transcriptions = ["" for _ in regions]
        for line in pagexml.lines_iterator():
            line_bbox = polygon_to_bbox(line.polygon)

            idx = best_intersecting_bbox(line_bbox, regions)
            if idx is not None and line and line.transcription and line.transcription.strip():
                transcriptions[idx] += f"{line.transcription.strip()}\n"

        return transcriptions
    
    def get_mapping(self, regions: List[AABB], document: AnnotatedDocument) -> Dict[int, Optional[str]]:
        mapping = {}

        for i, yolo_region in enumerate(regions):
            idx = best_intersecting_bbox(yolo_region, [r.bbox for r in document.regions])
            mapping[i] = None if idx is None else document.regions[idx].id

        return mapping
    
    def create_all_edges(self, regions: List[AABB]) -> List[Tuple[int, int]]:
        edges = list(itertools.combinations(range(len(regions)), 2))
        edges = edges + [(to_idx, from_idx) for from_idx, to_idx in edges]
        return edges
    
    def create_geometric_edges(self, regions: List[AABB], pagexml: PageLayout) -> List[Tuple[int, int]]:
        geometry = PageGeometry(regions=regions, pagexml=pagexml)
        geometry.set_region_visibility()

        edges = []

        for from_idx, geometry_region in enumerate(geometry.regions):
            for visible_region in geometry_region.visible_entities:
                to_idx = geometry.regions.index(visible_region)
                edges.append((from_idx, to_idx))
                edges.append((to_idx, from_idx))

        edges = [(f, t) for f, t in edges if f != t]

        return list(set(edges))
    
    def create_labels(
        self,
        regions: List[AABB],
        edges: List[Tuple[int, int]],
        document: Optional[AnnotatedDocument],
        run_checks: bool=False,
    ) -> List[int]:
        labels = []

        if document is None:
            return labels

        mapping = self.get_mapping(regions, document)

        relations = [set([id, ids[0]]) for id, ids in document.relations.items()]

        for from_idx, to_idx in edges:
            from_id, to_id = mapping[from_idx], mapping[to_idx]
            label = int(set([from_id, to_id]) in relations)
            labels.append(label)

        # Length check
        assert len(labels) == len(edges)

        # Fix transitivity
        positive_edges = [edge for label, edge in zip(labels, edges) if label == 1]
        positive_subsets = []
        for from_idx, to_idx in positive_edges:
            subset = [(from_idx, to_idx)]
            while True:
                from_indices = [f for f, _ in subset]
                to_indices = [t for _, t in subset]
                left_positive_edges = [(f, t) for f, t in positive_edges if t in from_indices and (f, t) not in subset]
                right_positive_edges = [(f, t) for f, t in positive_edges if f in to_indices and (f, t) not in subset]

                if not left_positive_edges and not right_positive_edges:
                    break

                subset.extend(left_positive_edges)
                subset.extend(right_positive_edges)

            subset = set(subset)
            if subset not in positive_subsets:
                positive_subsets.append(set(subset))

        for ps in positive_subsets:
            indices = [f for f, _ in ps] + [t for _, t in ps]
            indices = list(set(indices))
            for edge in itertools.permutations(indices, 2):
                try:
                    idx = edges.index(edge)
                except ValueError:
                    continue
                labels[idx] = 1

        # Symmetry check
        if run_checks:
            for label, (from_idx, to_idx) in zip(labels, edges):
                reverse_edge = (to_idx, from_idx)
                reverse_edge_idx = edges.index(reverse_edge)
                reverse_edge_label = labels[reverse_edge_idx]
                assert label == reverse_edge_label

        return labels
    
    def create_edge_attr(
            self,
            edges: List[Tuple[int, int]],
            geometry: PageGeometry,
            czert_embeddings: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
        ) -> List[torch.FloatTensor]:
        edge_attrs = []

        for from_idx, to_idx in edges:
            edge_attr = []

            from_region = geometry.regions[from_idx]
            to_region = geometry.regions[to_idx]

            x_dist = abs(from_region.center.x - to_region.center.x)
            y_dist = abs(from_region.center.y - to_region.center.y)
            dist = dist_l2(from_region.center, to_region.center)

            edge_attr.append(x_dist)
            edge_attr.append(y_dist)
            edge_attr.append(dist)

            edge_attr.append(x_dist / geometry.page_width)
            edge_attr.append(y_dist / geometry.page_height)

            distance_x = max(0, min(to_region.bbox.xmin - from_region.bbox.xmax, to_region.bbox.xmax - from_region.bbox.xmin))
            distance_y = max(0, min(to_region.bbox.ymin - from_region.bbox.ymax, to_region.bbox.ymax - from_region.bbox.ymin))

            edge_attr.append(distance_x)
            edge_attr.append(distance_y)
            edge_attr.append(distance_x / geometry.page_width)
            edge_attr.append(distance_y / geometry.page_height)

            from_pooler, from_cls = czert_embeddings[from_idx]
            to_pooler, to_cls = czert_embeddings[to_idx]

            pooler_dist = torch.cosine_similarity(from_pooler, to_pooler).item()
            cls_dist = torch.cosine_similarity(from_cls, to_cls).item()

            edge_attr.append(pooler_dist)
            edge_attr.append(cls_dist)

            if (from_region.child and from_region.child is to_region) or \
               (from_region.parent and from_region.parent is to_region) or \
               (to_region.child and to_region.child is from_region) or \
               (to_region.parent and to_region.parent is from_region):
                edge_attr.append(1.0)
            else:
                edge_attr.append(0.0)

            edge_attrs.append(torch.tensor(edge_attr, dtype=torch.float32))

        return edge_attrs

    def get_graph_from_bites(
        self,
        bites: List[Bite],
        filename: str,
        pagexml: PageLayout,
        document: Optional[AnnotatedDocument]=None, # None when running inference
    ) -> Graph:
        all_regions = [bite.bbox for bite in bites]

        if len(all_regions) < 2:
            raise RuntimeError("To create graph from regions, at least 2 regions must exist")

        transcriptions = self.get_transcriptions(all_regions, pagexml)
        geometry = PageGeometry(regions=all_regions, pagexml=pagexml)
        czert_embeddings = []

        for t in transcriptions:
            czert_embedding = self.text_features_provider.get_czert_features(t)
            czert_embeddings.append(czert_embedding)

        node_features = self.geometric_features_provider.get_regions_features(geometry, pagexml)

        # edges = self.create_geometric_edges(all_regions, pagexml)
        edges = self.create_all_edges(all_regions)

        edge_attr = self.create_edge_attr(edges, geometry, czert_embeddings)

        labels = self.create_labels(all_regions, edges, document)

        from_indices = [from_idx for from_idx, _ in edges]
        to_indices = [to_idx for _, to_idx in edges]

        graph = Graph(
            id=filename,
            node_features=node_features,
            from_indices=from_indices,
            to_indices=to_indices,
            edge_attr=edge_attr,
            labels=labels,
        )

        return graph


if __name__ == "__main__":
    from ultralytics import YOLO
    from textbite.data_processing.label_studio import LabelStudioExport
    import logging
    from textbite.utils import CZERT_PATH

    MODEL_PATH = r"/home/martin/textbite/yolo/yolo-models-200/yolo-s-800.pt"

    # FILENAME = "hudba-vecneho-zivota-02"
    # IMG_PATH = r"/home/martin/textbite/data/segmentation/images/val-book/{FILENAME}.jpg"
    # XML_PATH = r"/home/martin/textbite/data/segmentation/xmls/val-book/{FILENAME}.xml"

    # FILENAME = "hudba-vecneho-zivota-02"
    # IMG_PATH = f"/home/martin/textbite/data/segmentation/images/val-book/{FILENAME}.jpg"
    # XML_PATH = f"/home/martin/textbite/data/segmentation/xmls/val-book/{FILENAME}.xml"

    FILENAME = "cesky-zapad-(1)-1699212718-2"
    IMG_PATH = f"/home/martin/textbite/data/segmentation/images/val-peri/{FILENAME}.jpg"
    XML_PATH = f"/home/martin/textbite/data/segmentation/xmls/val-peri/{FILENAME}.xml"
    
    EXPORT_PATH = r"/home/martin/textbite/data/segmentation/export-3396-22-01-2024.json"

    # Load CZERT and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Creating graphs on: {device}")

    logging.info("Loading tokenizer ...")
    tokenizer = BertTokenizerFast.from_pretrained(CZERT_PATH)
    logging.info("Tokenizer loaded.")
    logging.info("Loading CZERT ...")
    czert = BertModel.from_pretrained(CZERT_PATH)
    czert = czert.to(device)
    logging.info("CZERT loaded.")
    
    graph_provider = JoinerGraphProvider(tokenizer, czert, device)
    biter = YoloBiter(YOLO(MODEL_PATH))
    pagexml = PageLayout(file=XML_PATH)
    export = LabelStudioExport(path=EXPORT_PATH)
    doc = export.documents_dict[f"{FILENAME}.jpg"]

    bites = biter.produce_bites(IMG_PATH, pagexml)

    graph = graph_provider.get_graph_from_file(
        bites=bites,
        path_img=IMG_PATH,
        pagexml=pagexml,
        document=doc,
    )

    asdf = graph.flatten()
    # print(graph)
    # print(graph.node_features)
    # print(f"Edges: {graph.edge_index}")
    # print(f"Labels: {graph.labels}")
