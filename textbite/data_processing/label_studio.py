"""Definition of the export data file from Label-Studio. Running this file is for debug only.

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


from typing import Optional, List, Tuple, Dict
import json
from dataclasses import dataclass, field
from enum import Enum
import json
import urllib.parse
import logging

from pero_ocr.document_ocr.layout import PageLayout, TextLine

from textbite.geometry import AABB, polygon_to_bbox, bbox_to_yolo, best_intersecting_bbox


class RegionType(Enum):
    TITLE = "title"
    TEXT = "text"
    NOTE = "note"
    IMAGE = "image"
    IMAGE_DESC = "image-desc"


class DocumentType(Enum):
    BOOK = "book"
    DICTIONARY = "dictionary"
    PERIODICAL = "periodical"


@dataclass
class AnnotatedRegion:
    id: str
    label: RegionType
    bbox: AABB
    lines: List[TextLine] = field(default_factory=list)


@dataclass
class AnnotatedDocument:
    id: int
    filename: str
    document_type: DocumentType
    regions: List[AnnotatedRegion]
    relations: Dict[str, List[str]]
    width: float
    height: float
    mapped_to_pagexml: bool=False

    def map_to_pagexml(self, pagexml: PageLayout) -> None:
        for line in pagexml.lines_iterator():
            if line and line.transcription and line.transcription.strip():
                self.map_line(line)
        self.regions = [region for region in self.regions if region.lines]

        h, w = pagexml.page_size
        self.width = w
        self.height = h

        self.map_to_pagexml = True

    def map_line(self, line: TextLine) -> None:
        line_bbox = polygon_to_bbox(line.polygon)

        best_region_idx = best_intersecting_bbox(line_bbox, [r.bbox for r in self.regions])
        if best_region_idx is not None:
            best_region = self.regions[best_region_idx]
            best_region.lines.append(line)

    def to_json(self) -> List[List[str]]:
        return [[line.id for line in region.lines] for region in self.regions]

    def get_json_str(self, indent: Optional[int]=None) -> str:
        return json.dumps(self.to_json(), indent=indent)

    def to_yolo(self) -> List[Tuple[int, float, float, float, float]]:
        yolos = []

        if not self.mapped_to_pagexml:
            for region in self.regions:
                match region.label:
                    case RegionType.TITLE:
                        label = 0

                    case RegionType.TEXT:
                        label = 1

                    case _:
                        continue

                yolos.append((label,) + bbox_to_yolo(region.bbox, self.width, self.height))
            
            return yolos
    
        for region in self.regions:
            bboxes = [polygon_to_bbox(line.polygon) for line in region.lines]
            min_x = min(bboxes, key=lambda x: x.xmin).xmin
            min_y = min(bboxes, key=lambda x: x.ymin).ymin
            max_x = max(bboxes, key=lambda x: x.xmax).xmax
            max_y = max(bboxes, key=lambda x: x.ymax).ymax
            bbox = AABB(min_x, min_y, max_x, max_y)

            match region.label:
                case RegionType.TITLE:
                    label = 3

                case RegionType.TEXT:
                    label = 0
                    first = min(bboxes, key=lambda x: x.ymin)
                    last = max(bboxes, key=lambda x: x.ymax)

                    yolos.append((1,) + bbox_to_yolo(first, self.width, self.height))
                    yolos.append((2,) + bbox_to_yolo(last, self.width, self.height))

                case _:
                    continue

            yolos.append((label,) + bbox_to_yolo(bbox, self.width, self.height))

        return yolos

    def get_yolo_str(self) -> str:
        yolo_str = ""

        for label, x, y, w, h in self.to_yolo():
            yolo_str += f"{label} {x} {y} {w} {h}\n"

        return yolo_str

    def merge_regions(self):
        translations = {r.id: r.id for r in self.regions}
        regions_dict = {r.id: r for r in self.regions}

        for src, dst in self.relations.items():
            if len(dst) > 0:
                # logging.warning(f'There are multiple outgoing relations for {src}')
                pass

            dst = dst[0]
            if src in translations and dst in translations:
                true_src = translations[src]
                true_dst = translations[dst]

                try:
                    regions_dict[true_src].lines.extend(regions_dict[true_dst].lines)
                except KeyError:
                    continue
                translations[true_dst] = true_src

                del regions_dict[true_dst]
            else:
                if src in translations:
                    translations[dst] = translations[src]
                elif dst in translations:
                    translations[src] = translations[dst]
                else:
                    logging.warning(f'Relation between two nonexistent regions!')

        self.regions = list(regions_dict.values())


class LabelStudioExport:
    def __init__(self, path: Optional[str]=None, raw_data: Optional[List[dict]]=None):
        if path:
            with open(path, "r") as f:
                raw_data = json.load(f)

        self.documents: List[AnnotatedDocument] = []
        self.parse_json_str(raw_data=raw_data)
        self.documents_dict = {doc.filename: doc for doc in self.documents}

    def __len__(self):
        return len(self.documents)

    def parse_json_str(self, raw_data: List[dict]) -> None:
        for annotated_file in raw_data:
            id = int(annotated_file["id"])
            annotated_regions = annotated_file["annotations"]
            pth = annotated_file["data"]["image"]

            partition = pth.rpartition("/")
            filename = urllib.parse.unquote(partition[2])

            document_type = partition[0].rpartition("/")[2]
            document_type = document_type.partition("-")[0]
            document_type = DocumentType(document_type)

            try:
                regions, relations = self.parse_annotated_regions(annotated_regions)
            except ValueError:
                continue

            document = AnnotatedDocument(
                id=id,
                filename=filename,
                document_type=document_type,
                regions=regions,
                relations=relations,
                width=annotated_regions[0]["result"][0]["original_width"],
                height=annotated_regions[0]["result"][0]["original_height"],
            )
            self.documents.append(document)

    def parse_annotated_regions(self, annotated_regions: List[dict]) -> Tuple[List[AnnotatedRegion], Dict[str, List[str]]]:
        if len(annotated_regions) > 1:
            logging.warning(f"Multiple annotations detected")
            raise ValueError

        _annotated_regions = annotated_regions[0]["result"]

        relations = [r for r in _annotated_regions if r["type"] == "relation"]
        regions = [r for r in _annotated_regions if r["type"] == "rectanglelabels"]

        relations_parsed = self.parse_relations(relations)
        regions_parsed = self.parse_regions(regions)
        return regions_parsed, relations_parsed

    def parse_relations(self, relations: List[dict]) -> Dict[str, List[str]]:
        relations_dict = {relation["from_id"]: [] for relation in relations}
        for relation in relations:
            relations_dict[relation["from_id"]].append(relation["to_id"])
        return relations_dict

    def parse_regions(self, regions: List[dict]) -> List[AnnotatedRegion]:
        result = []
        for region in regions:
            id = region["id"]
            label = RegionType(region["value"]["rectanglelabels"][0])
            bbox = LabelStudioExport.get_bbox(region["value"], region["original_width"], region["original_height"])

            region_parsed = AnnotatedRegion(id, label, bbox)
            result.append(region_parsed)

        return result

    def get_bbox(annotation: dict, page_width: int, page_height: int) -> AABB:
        xmin = (annotation["x"] / 100.0) * page_width
        ymin = (annotation["y"] / 100.0) * page_height

        region_width = (annotation["width"] / 100.0) * page_width
        region_height = (annotation["height"] / 100.0) * page_height

        xmax = xmin + region_width
        ymax = ymin + region_height

        return AABB(xmin, ymin, xmax, ymax)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)

    EXPORT_PATH = r"/home/martin/textbite/data/segmentation/export-3397-22-01-2024.json"
    PAGEXML_PATH = r"/home/martin/textbite/data/segmentation/xmls/val-book/nikola-suhaj-loupeznik-06.xml"
    with open(EXPORT_PATH, "r") as f:
        raw_data = json.load(f)
    export1 = LabelStudioExport(raw_data=raw_data)
    print(len(export1))
    # for doc in export1.documents:
    #     if "nikola-suhaj-loupeznik-06" in doc.filename:
    #         pagexml = PageLayout(file=PAGEXML_PATH)
    #         doc.map_to_pagexml(pagexml)
    #         # print([region.lines for region in doc.regions])
    #         break