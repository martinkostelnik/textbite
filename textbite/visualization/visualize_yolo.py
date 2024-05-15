"""Visualize hand annotated bites in YOLO format.
   Will visualize titles in WHITE.
   Will visualize regions as: first line RED, whole region GREEN, last line BLUE.

    Date -- 02.12.2023
    Author -- Martin Kostelnik
"""


import sys
import argparse
import os
import logging

import cv2


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--yolo", required=True, type=str, help="Path to a folder with yolo files.")
    parser.add_argument('--images', required=True, type=str, help="Path to a folder with image files.")
    parser.add_argument("--save", required=True, type=str, help="Path to a folder where results will be saved.")

    args = parser.parse_args()
    return args


def draw_bboxes(image, lines):
    h, w, _ = image.shape
    
    for line in lines:
        label, x_center, y_center, width, height = line.split()
        label = int(label)
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)

        if label == 0: # Text GREEN 
            color = (0, 255, 0)
        elif label == 1: # First line RED
            color = (0, 0, 255)
        elif label == 2: # Last line BLUE
            color = (255, 0, 0)
        else: # Title WHITE
            color = (255, 255, 255)

        # Draw center point
        x = int(x_center * w)
        y = int(y_center * h)
        cv2.circle(image, (x, y), 15, color, 10)
            
        # Draw bounding box
        start_x = int((x_center - (width / 2)) * w)
        start_y = int((y_center - (height / 2)) * h)
        end_x = int((x_center + (width / 2)) * w)
        end_y = int((y_center + (height / 2)) * h)
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 6)
    return image


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level,force=True)
    os.makedirs(args.save, exist_ok=True)
    
    yolo_files = [filename for filename in os.listdir(args.yolo) if filename.endswith(".yolo")]
    
    for yolo_file in yolo_files:
        img_filename = yolo_file.replace(".yolo", ".jpg")
        img_path = os.path.join(args.images, img_filename)
        save_path = os.path.join(args.save, img_filename.replace(".jpg", "-bboxes.jpg"))
        yolo_path = os.path.join(args.yolo, yolo_file)

        logging.debug(f"Loading | {yolo_path} | {img_path} | {save_path}")
        image = cv2.imread(img_path)

        with open(yolo_path, "r") as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines if line.strip()]
        if not lines:
            continue

        image_bboxes = draw_bboxes(image, lines)
        cv2.imwrite(save_path, image_bboxes)


if __name__ == "__main__":
    main()
