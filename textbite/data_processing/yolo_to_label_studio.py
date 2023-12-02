import os
import cv2
import json
import argparse
from tqdm import tqdm


WANTED_LABELS = ["obrázek", "fotografie"]


# OLD_MAPPING = {
#     '0': "Erb/arms/logo",
#     '1': "Graph",
#     '2': "Image",
#     '3': "Initial",
#     '4': "Map",
#     '5': "Math formula",
#     '6': "Music notation",
#     '7': "Separator",
#     '8': "Stamp",
#     '9': "Table",
# }

# NEW_MAPPING = {
#     '0': "Erb/cejch/logo/symbol",
#     '1': "Graf",
#     '2': "obrázek",
#     '3': "Iniciála",
#     '4': "Mapa",
#     '5': "Matematická formule",
#     '6': "Notový zápis",
#     '7': "Ostatní knižní dekor",
#     '8': "Razítko",
#     '9': "Tabulka",
# }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help="Path to the images directory.", required=True)
    parser.add_argument('--labels', help="Path to the YOLO labels directory.", required=True)
    parser.add_argument('--output', help="Path to the output file.", required=True)
    return parser .parse_args()


def load_image(path):
    return cv2.imread(path)


def load_labels(path):
    data = []

    if os.path.exists(path):
        with open(path) as file:
            for line in file:
                line = line.strip()
                if len(line) > 0:
                    line_parts = line.split()
                    label = (line_parts[0], float(line_parts[1]), float(line_parts[2]), float(line_parts[3]), float(line_parts[4]))
                    data.append(label)

    return data


def convert(labels, image, image_name):
    result = {
        "data": { "image": f"/data/local-files/?d=orbis-pictus/images/{image_name}" },
        "predictions": [{ "model_version": "v0.1", "score": "1.0", "result": []}]
    }

    image_height, image_width = image.shape[:2]

    for i, label in enumerate(labels):
        # import IPython; IPython.embed()

        label_class, label_x, label_y, label_width, label_height = label

        if label_class not in WANTED_LABELS:
            continue

        x = label_x / image_width
        x *= 100

        y = label_y / image_height
        y *= 100

        w = (label_width - label_x) / image_width
        w *= 100

        h = (label_height - label_y) / image_height
        h *= 100

        result["predictions"][0]["result"].append({
            "id": f"result{i}",
            "type": "rectanglelabels",
            "from_name": "label", "to_name": "image",
            "original_width": image_width, "original_height": image_height,
            "image_rotation": 0,
            "value": {
                "rotation": 0,          
                "x": x, "y": y,
                "width": w, "height": h,
                "rectanglelabels": [label_class]
            }
        })

    return result


def save(data, path):
    for f in data:
        filename = os.path.basename(f["data"]["image"]).replace(".jpg", ".json")
        with open(os.path.join(path, filename), 'w') as file:
            json.dump(f, file, indent=4)


def main(args):
    files = [file for file in os.listdir(args.images) if file.endswith(".jpg")]

    data = []

    for image_file in tqdm(files):
        label_file = image_file.replace(".jpg", ".txt")

        image_path = os.path.join(args.images, image_file)
        labels_path = os.path.join(args.labels, label_file)

        image = load_image(image_path)
        labels = load_labels(labels_path)

        labels = convert(labels, image, image_file)

        data.append(labels)

    save(data, args.output)

    return 0


if __name__ == "__main__":
    exit(main(parse_args()))
