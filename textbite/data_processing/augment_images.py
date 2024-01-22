import sys
import os
import argparse

import cv2
import numpy as np
import random


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--images", required=True, type=str, help="Path to a folder containing images.")
    parser.add_argument("--save", default=".", type=str, help="Where to store results.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    os.makedirs(args.save, exist_ok=True)

    img_filenames = [img_filename for img_filename in os.listdir(args.images) if img_filename.endswith(".jpg")]

    for img_filename in img_filenames:
        img_path = os.path.join(args.images, img_filename)

        image = cv2.imread(img_path)
        rows, cols = image.shape[:2]

        # Color distortion
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        h += np.random.randint(0, 100,size=(rows, cols), dtype=np.uint8 )
        s += np.random.randint(0, 20,size=(rows, cols), dtype=np.uint8 )
        v += np.random.randint(0, 10,size=(rows, cols) , dtype=np.uint8 )
        distorted_image = cv2.merge([h, s, v])
        distorted_image = cv2.cvtColor(distorted_image, cv2.COLOR_HSV2BGR)
        save_path = os.path.join(args.save, img_filename.replace(".jpg", "-distorted.jpg"))
        cv2.imwrite(save_path, distorted_image)

        # Blur
        blur_val = random.randint(10, 20) #blur value random
        blurred_image = cv2.blur(image,(blur_val, blur_val))
        save_path = os.path.join(args.save, img_filename.replace(".jpg", "-blurred.jpg"))
        cv2.imwrite(save_path, blurred_image)

        # Increase brightness
        increase = np.ones(image.shape, dtype="uint8") * 70
        brightened_image = cv2.add(image, increase)
        save_path = os.path.join(args.save, img_filename.replace(".jpg", "-bright.jpg"))
        cv2.imwrite(save_path, brightened_image)

        # Decrease brightness
        decrease = np.ones(image.shape, dtype="uint8") * 240
        darkened_image = cv2.subtract(image, increase)
        save_path = os.path.join(args.save, img_filename.replace(".jpg", "-dark.jpg"))
        cv2.imwrite(save_path, darkened_image)


if __name__ == "__main__":
    main()
