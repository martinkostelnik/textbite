import logging
import argparse
import sys

from safe_gpu import safe_gpu


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", required=True, type=str, help="Path to a pickle file with training data.")

    args = parser.parse_args()
    return args


def main():
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        force=True,
    )
    args = parse_arguments()
    safe_gpu.claim_gpus()
    main(args)
