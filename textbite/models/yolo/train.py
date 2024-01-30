import argparse

from ultralytics import YOLO

from safe_gpu import safe_gpu


def parse_aguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, type=str, help="Path to the .pt file with pretrained YOLO model")
    parser.add_argument("--data", required=True, type=str, help="Path to the .yaml file with data")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imsize", "-i", type=int, default=640, help="Image size")
    parser.add_argument("--save", required=True, type=str, help="Folder where to save the outputs")

    return parser.parse_args()


def main() -> None:
    safe_gpu.claim_gpus()
    args = parse_aguments()

    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imsize)


if __name__ == "__main__":
    main()
