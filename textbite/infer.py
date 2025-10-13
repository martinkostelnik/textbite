import argparse
import os
import logging
import pickle
from tqdm import tqdm

from numba.core.errors import NumbaDeprecationWarning
import warnings
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

from safe_gpu import safe_gpu
from pero_ocr.core.layout import PageLayout
from ultralytics import YOLO
import torch
from transformers import BertTokenizerFast, BertModel
import numpy as np
import cv2

from textbite.utils import CZERT_PATH
from textbite.models.yolo.infer import YoloBiter
from textbite.models.utils import GraphNormalizer
from textbite.models.joiner.graph import JoinerGraphProvider
from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.joiner.infer import join_bites
from textbite.models.improve_pagexml import PageXMLEnhancer, UnsupportedLayoutError
from textbite.bite import save_bites


IMAGE_EXTENSIONS = [".jpg", ".jpeg"]
CLASSIFICATION_THRESHOLD = 0.5


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])

    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--img", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--alto", default=None, type=str, help="Path to a folder with alto data.")

    parser.add_argument("--yolo", required=True, type=str, help="Path to the .pt file with weights of YOLO model.")
    parser.add_argument("--gnn", required=True, type=str, help="Path to the .pt file with weights of Joiner model.")
    parser.add_argument("--normalizer", required=True, type=str, help="Path to node normalizer.")
    parser.add_argument("--czert", default=CZERT_PATH, type=str, help="Path to CZERT.")

    parser.add_argument("--json", action="store_true", help="Store the JSON output format.")
    parser.add_argument("--render", action="store_true", help="Render the output images.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output files.")

    return parser.parse_args()


COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (34, 139, 34),    # Forest Green
    (70, 130, 180),   # Steel Blue
    (255, 20, 147),   # Deep Pink
    (218, 112, 214),  # Orchid
    (255, 165, 0),    # Orange
    (173, 216, 230),  # Light Blue
    (255, 69, 0),     # Red-Orange
    (0, 191, 255),    # Deep Sky Blue
    (128, 0, 128),    # Purple
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 99, 71),    # Tomato
    (255, 192, 203),  # Pink
    (32, 178, 170),   # Light Sea Green
    (250, 128, 114),  # Salmon
    (0, 128, 128),    # Teal
    (240, 230, 140)   # Khaki
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize results of textbite runs.")

    parser.add_argument(
        "-o:",
        "--output",
        type=str,
        required=True,
        help="Output folder for the visualized results.",
    )

    parser.add_argument(
        "-n:",
        "--n-documents",
        type=int,
        default=10,
        help="Number of documents to visualize from each library.",
    )

    return parser.parse_args()


def overlay_line(img, line, color, alpha):
    mask = np.zeros_like(img)
    pts = line.polygon
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color)
    return cv2.addWeighted(img, 1, mask, 1-alpha, 0)


def render_bites(
    pagexml: PageLayout,
    image: np.ndarray,
    bites,
) -> np.ndarray:
    overlay = np.zeros_like(image)
    ALPHA = 0.4

    lines_dict = {line.id: line for line in pagexml.lines_iterator()}

    for i, bite in enumerate(bites):
        color = COLORS[i % len(COLORS)]

        for line_id in bite.lines:
            line = lines_dict.get(line_id)
            if not line.transcription.strip():
                continue

            if line.id not in lines_dict:
                logging.warning(f"Line {line.id} not found in pagexml.")
                continue

            overlay = overlay_line(overlay, line, color, ALPHA)

    return cv2.addWeighted(image, 1, overlay, 1-ALPHA, 0)


def visualize_predictions(
    pagexml: PageLayout,
    image_path: str,
    bites,
    yolo_bites,
) -> np.ndarray:
    image = cv2.imread(image_path)
    rendered_bites = render_bites(pagexml, image, bites)
    rendered_yolo_bites = render_bites(pagexml, image, yolo_bites)

    combined_image = cv2.hconcat([image, rendered_yolo_bites, rendered_bites])
    return combined_image


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    safe_gpu.claim_gpus()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YoloBiter(YOLO(args.yolo))

    tokenizer = BertTokenizerFast.from_pretrained(args.czert)
    czert = BertModel.from_pretrained(args.czert)
    czert = czert.to(device)

    graph_provider = JoinerGraphProvider(tokenizer, czert, device)

    gnn_checkpoint = torch.load(args.gnn)
    gnn = JoinerGraphModel.from_pretrained(gnn_checkpoint, device)
    gnn.eval()
    gnn = gnn.to(device)

    with open(args.normalizer, "rb") as f:
        normalizer: GraphNormalizer = pickle.load(f)

    xml_enhancer = PageXMLEnhancer()

    os.makedirs(args.save, exist_ok=True)

    img_filenames = [img_filename for img_filename in os.listdir(args.img) if os.path.splitext(img_filename)[1] in IMAGE_EXTENSIONS]
    for i, img_filename in tqdm(enumerate(img_filenames)):
        # if "b1576790-3323-11ec-a678-001b63bd97ba" not in img_filename:
        #     continue
        img_extension = os.path.splitext(img_filename)[1]
        xml_filename = img_filename.replace(img_extension, ".xml")
        base_filename = xml_filename.replace(".xml", "")
        json_filename = xml_filename.replace(".xml", ".json")

        img_path = os.path.join(args.img, img_filename)
        xml_path = os.path.join(args.xml, xml_filename)
        alto_path = os.path.join(args.alto, xml_filename) if args.alto is not None else None
        json_save_path = os.path.join(args.save, json_filename)
        xml_save_path = os.path.join(args.save, xml_filename)

        try:
            pagexml = PageLayout(file=xml_path)
        except OSError:
            logging.warning(f"XML file {xml_path} not found. SKIPPING")
            continue

        logging.info(f"({i+1}/{len(img_filenames)}) | Processing: {xml_path}")

        yolo_bites = yolo.produce_bites(img_path, pagexml, alto_path)

        try:
            bites = join_bites(
                yolo_bites,
                gnn,
                graph_provider,
                normalizer,
                base_filename,
                pagexml,
                device,
                CLASSIFICATION_THRESHOLD,
            )
        except RuntimeError:
            logging.info(f"Single region detected on {xml_path}, saving as is.")
            bites = yolo_bites

        # try:
        #     out_xml_string = xml_enhancer.process(pagexml, bites)
        #     with open(xml_save_path, 'w', encoding='utf-8') as f:
        #         print(out_xml_string, file=f)
        # except UnsupportedLayoutError as e:
        #     logging.warning(e)

        if args.json:
            save_bites(bites, json_save_path)
    
        if args.render:
            rendered_predictions = visualize_predictions(
                pagexml,
                img_path,
                bites,
                yolo_bites,
            )
            output_path_img = os.path.join(args.save, f"{base_filename}.jpg")
            cv2.imwrite(output_path_img, rendered_predictions)



if __name__ == "__main__":
    main()
