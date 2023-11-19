import argparse
import os
import sys
import json

from pero_ocr.document_ocr.layout import PageLayout


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, type=str, help="Path to a folder containing xml data.")
    parser.add_argument("--out", required=True, type=str, help="Path to a folder to save output.")

    args = parser.parse_args()
    return args


def transform_points(points, width, height):
    for point in points:
        point[0] = point[0] / width * 100.0
        point[1] = point[1] / height * 100.0
        
        point[0] = 100.0 if point[0] > 100.0 else point[0]
        point[1] = 100.0 if point[1] > 100.0 else point[1]
        
    return points


def main(args):
    for filename in os.listdir(args.data):
        if not filename.endswith(".xml"):
            continue

        path = os.path.join(args.data, filename)
        pagexml = PageLayout(file=path)
        result = {
            "data": {"image": f"/data/local-files/?d=textbite/{filename[:-3]}jpg"},
            "predictions": [{"result": []}]}

        original_height, original_width = pagexml.page_size

        for id, region in enumerate(pagexml.regions):
            item = {
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "id": f"{id}",
                "from_name": "polygon",
                "to_name": "image",
                "type": "polygon",
                "value": {"points": transform_points(region.polygon.tolist(), original_width, original_height)},
            }

            result["predictions"][0]["result"].append(item)
            result_str = json.dumps(result, indent=2)

        out_path = os.path.join(args.out, f"{filename[:-3]}json")
        with open(out_path, "w") as f:
            print(result_str, file=f)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
