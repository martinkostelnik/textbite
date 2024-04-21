import argparse
import logging
from time import perf_counter
import os

from ultralytics import YOLO

from safe_gpu import safe_gpu


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--model", required=True, type=str, help="Path to the .pt file with weights of YOLO model.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("safe_gpu").setLevel(logging.WARNING)
    safe_gpu.claim_gpus()

    model = YOLO(args.model)

    img_filenames = [img_filename for img_filename in os.listdir(args.images) if img_filename.endswith(".jpg")]
    n_files = len(img_filenames)
    total_time = 0.0

    for img_filename in img_filenames:
        path_img = os.path.join(args.images, img_filename)

        start = perf_counter()
        model.predict(source=path_img)
        end = perf_counter()

        t = end - start
        logging.debug(f"Image {repr(path_img)} took: {t} s")
        total_time += t

    logging.info(f"{args.model}")
    logging.info(f"Total time for {n_files} images: {total_time:.4f} s | Avg time per page: {(total_time/n_files):.4f} s.\n")


if __name__ == "__main__":
    main()
