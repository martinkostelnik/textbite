from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass
import logging
import os
import random
from tqdm import tqdm

import numpy as np
import torch
from pero_ocr.core.layout import PageLayout

from textbite.geometry import polygon_to_bbox

logger = logging.getLogger(__name__)


@dataclass
class Paragraph:
    id: int
    transcription: str
    bbox: list[float]  # [x, y, w, h]
    embedding: np.ndarray  # (D,)
    successor: Paragraph | None
    bite_id: int


class TextBiteContrastiveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_path: Path,
        text_embedding_dir: Path,
        data_limit: int | None = None,
    ):
        super().__init__()
        
        self.coco_path = coco_path
        self.text_embedding_dir = text_embedding_dir
        self.data_limit = data_limit

        self.pairs = self.load()

    def __len__(
        self,
    ):
        return len(self.pairs)
    
    def __getitem__(
        self,
        idx: int,
    ):
        idx = idx % len(self)

        paragraph_a, paragraph_b, label = self.pairs[idx]
        return {
            "paragraph_a_bbox": paragraph_a.bbox,
            "paragraph_a_text_embedding": paragraph_a.embedding,
            "paragraph_b_bbox": paragraph_b.bbox,
            "paragraph_b_text_embedding": paragraph_b.embedding,
            "label": label,
        }

    def load(
        self,
    ) -> tuple[Paragraph, Paragraph, int]:
        logger.debug(f"Loading dataset from {self.coco_path} and {self.text_embedding_dir} ...")

        with open(self.coco_path, "r") as f:
            coco_data = json.load(f)

        logger.debug(f"Loaded {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations, and {len(coco_data['relations'])} relations.")

        paragraphs = []

        sample_iter = coco_data["images"] if self.data_limit is None or self.data_limit > len(coco_data["images"]) else coco_data["images"][:self.data_limit]
        for annotated_sample in tqdm(sample_iter, desc="Loading samples"):
            sample_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == annotated_sample["id"]]
            sample_relations = [rel for rel in coco_data["relations"] if rel["image_id"] == annotated_sample["id"]]

            image_width = annotated_sample["width"]
            image_height = annotated_sample["height"]
            
            sample_text_embeddings = np.load(self.text_embedding_dir / f"{annotated_sample['file_name'][:-4]}.npy").astype(np.float32)
            sample_text_embeddings = sample_text_embeddings / np.linalg.norm(sample_text_embeddings, axis=1, keepdims=True)

            sample_paragraphs = []

            for i, ann in enumerate(sample_annotations):
                ann["embedding"] = sample_text_embeddings[i]

                xmin, ymin, w, h = ann["bbox"]
                cx = (xmin + w / 2) / image_width
                cy = (ymin + h / 2) / image_height
                w = w / image_width
                h = h / image_height

                paragraph = Paragraph(
                    id=ann["id"],
                    transcription=ann["transcription"],
                    bbox=np.array([cx, cy, w, h], dtype=np.float32),
                    embedding=ann["embedding"],
                    successor=None,
                    bite_id=i,
                )
                sample_paragraphs.append(paragraph)

            for rel in sample_relations:
                src_paragraph = next((p for p in sample_paragraphs if p.id == rel["from_annotation_id"]), None)
                tgt_paragraph = next((p for p in sample_paragraphs if p.id == rel["to_annotation_id"]), None)
                src_paragraph.successor = tgt_paragraph

            paragraphs.append(sample_paragraphs)

        logger.debug(f"Created {len(paragraphs)} samples with paragraphs.")

        self.group_paragraphs(paragraphs)

        positive_pairs = []
        negative_pairs = []

        for sample in paragraphs:
            for i, paragraph in enumerate(sample):
                for j, other_paragraph in enumerate(sample):
                    if i == j:
                        continue

                    if paragraph.bite_id == other_paragraph.bite_id:
                        positive_pairs.append((paragraph, other_paragraph, np.array(1, dtype=np.int64)))
                    else:
                        negative_pairs.append((paragraph, other_paragraph, np.array(-1, dtype=np.int64)))

        logger.debug(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs.")
        negative_pairs = random.sample(negative_pairs, k=len(positive_pairs))
        logger.debug(f"Balanced to {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs.")

        logging.info(f"Loaded {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs.")

        pairs = positive_pairs + negative_pairs
        return pairs

    def group_paragraphs(
        self,
        paragrahs: list[list[Paragraph]],
    ) -> None:
        for sample_paragraphs in paragrahs:
            succ_ptr = None
            for paragraph in sample_paragraphs:
                succ_ptr = paragraph.successor

                while succ_ptr is not None:
                    succ_ptr.bite_id = paragraph.bite_id
                    succ_ptr = succ_ptr.successor


class TextBitePageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        text_embedding_dir: Path,
        coco_path: Path | None = None,
        xml_dir: Path | None = None,
    ):
        super().__init__()

        self.coco_path = coco_path
        self.xml_dir = xml_dir
        self.text_embedding_dir = text_embedding_dir

        self.data = self.load()

    def __len__(
        self,
    ) -> int:
        return len(self.data)
    
    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:
        idx = idx % len(self)
        return self.data[idx]
    
    def load(
        self,
    ):
        if self.coco_path is not None:
            pages = self.load_from_coco()
        elif self.xml_dir is not None:
            pages = self.load_from_xml()
        else:
            raise ValueError("Either coco_path or xml_dir must be provided.")
        
        return pages
    
    def load_from_xml(
        self,
    ):
        pages = []
        xml_filenames = os.listdir(self.xml_dir)
        for xml_filename in xml_filenames:
            if not xml_filename.endswith(".xml"):
                continue

            xml_path = self.xml_dir / xml_filename
            text_embedding_path = self.text_embedding_dir / f"{xml_filename[:-4]}.npy"

            pagelayout = PageLayout(file=xml_path)
            text_embeddings = np.load(text_embedding_path).astype(np.float32)
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

            if len(pagelayout.regions) != len(text_embeddings):
                logger.warning(f"Number of regions in {xml_path} ({len(pagelayout.regions)}) does not match number of text embeddings ({len(text_embeddings)}). Skipping.")
                continue

            bboxes = []
            for region in pagelayout.regions:
                xmin, ymin, xmax, ymax = polygon_to_bbox(region.polygon)
                cx = (xmin + xmax) / 2 / pagelayout.width
                cy = (ymin + ymax) / 2 / pagelayout.height
                w = (xmax - xmin) / pagelayout.width
                h = (ymax - ymin) / pagelayout.height
                bbox = [cx, cy, w, h]
                bboxes.append(bbox)
            bboxes = np.array(bboxes, dtype=np.float32)
            pages.append({
                "file_name": xml_filename[:-4],
                "bbox": bboxes,
                "text_embedding": text_embeddings,
            })

        return pages

    def load_from_coco(
        self,
    ):
        with open(self.coco_path, "r") as f:
            coco_data = json.load(f)

        pages = []
        for annotated_sample in coco_data["images"]:
            sample_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == annotated_sample["id"]]

            image_width = annotated_sample["width"]
            image_height = annotated_sample["height"]
            
            text_embeddings_path = self.text_embedding_dir / f"{annotated_sample['file_name'][:-4]}.npy"
            text_embeddings = np.load(text_embeddings_path).astype(np.float32)
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

            if len(sample_annotations) != len(text_embeddings):
                logger.warning(f"Number of annotations in {self.coco_path} ({len(sample_annotations)}) does not match number of text embeddings ({len(text_embeddings)}). Skipping.")
                continue

            bboxes = []
            for i, ann in enumerate(sample_annotations):
                xmin, ymin, w, h = ann["bbox"]
                cx = (xmin + w / 2) / image_width
                cy = (ymin + h / 2) / image_height
                w = w / image_width
                h = h / image_height
                bbox = [cx, cy, w, h]
                bboxes.append(bbox)
            bboxes = np.array(bboxes, dtype=np.float32)
            pages.append({
                "file_name": annotated_sample["file_name"][:-4],
                "bbox": bboxes,
                "text_embedding": text_embeddings,
            })

        return pages