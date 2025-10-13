import argparse
import json
import os

import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize annotations')

    parser.add_argument('--coco', type=str, required=True, help='Path to COCO annotations file', nargs="+")
    parser.add_argument('--output', type=str, required=True, help='Path to output folder')
    parser.add_argument('--images', type=str, required=True, help='Path to images folder')

    return parser.parse_args()


def main():
    args = parse_arguments()

    os.makedirs(args.output, exist_ok=True)

    data = []
    for coco_file in args.coco:
        with open(coco_file, 'r') as f:
            data.append(json.load(f))

    image_filenames = os.listdir(args.images)
    for image_filename in image_filenames:
        image_path = os.path.join(args.images, image_filename)
        rendered_images = []

        for data_file in data:
            image_id = next((image['id'] for image in data_file['images'] if image['file_name'] == image_filename), None)
            if image_id is None:
                continue
            annotations = [annotation for annotation in data_file['annotations'] if annotation['image_id'] == image_id]
            relations = [relation for relation in data_file['relations'] if relation['image_id'] == image_id]

            image = cv2.imread(image_path)

            for annotation in annotations:
                bbox = annotation['bbox']
                x, y, w, h = bbox
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)


                match annotation["category_id"]:
                    case 0:
                        color = (0, 255, 0) # PAGE NUMBER GREEN
                    case 1:
                        color = (0, 0, 255) # TEXT RED
                    case 2:
                        color = (255, 0, 0) # TITLE BLUE
                    case _:
                        color = (255, 255, 0) # UNANNOTATED CYAN
                        continue

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)

                if "includes_title" in annotation:
                    if annotation["includes_title"]:
                        cv2.circle(image, (x + w // 2, y + h // 2), 10, (0, 255, 255), -1)

            for relation in relations:
                from_bbox = next((annotation['bbox'] for annotation in annotations if annotation['id'] == relation['from_annotation_id']), None)
                to_bbox = next((annotation['bbox'] for annotation in annotations if annotation['id'] == relation['to_annotation_id']), None)
                if to_bbox is None:
                    for a in annotations:
                        print(a)
                    print(f"{annotation=}")
                    print(f"{relation=}\n")
                from_x, from_y, from_w, from_h = from_bbox
                to_x, to_y, to_w, to_h = to_bbox

                from_x = int(from_x)
                from_y = int(from_y)
                from_w = int(from_w)
                from_h = int(from_h)
                to_x = int(to_x)
                to_y = int(to_y)
                to_w = int(to_w)
                to_h = int(to_h)

                from_center = (from_x + from_w // 2, from_y + from_h // 2)
                to_center = (to_x + to_w // 2, to_y + to_h // 2)

                cv2.arrowedLine(image, from_center, to_center, (0, 0, 255), 2)

            rendered_images.append(image)

        if not rendered_images:
            continue
        image = cv2.hconcat(rendered_images)
        output_path = os.path.join(args.output, "vis-" + image_filename)
        cv2.imwrite(output_path, image)


if __name__ == '__main__':
    main()
