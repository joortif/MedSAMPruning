import argparse

from datasets.download_datasets import download_dataset
from models.medsam import download_medsam

pruning_strategies = ["iou","cb-scs","el2n","entropy","forgetting","fusion"]

def build_parser():

    parser = argparse.ArgumentParser(
        description="Semantic segmentation dataset pruning"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ==========================================================
    # DOWNLOAD
    # ==========================================================

    download_parser = subparsers.add_parser(
        "download",
        help="Download datasets"
    )

    download_parser.add_argument(
        "--dataset",
        choices=["BUSI", "MMOTU"],
    )

    # ==========================================================
    # EXTRACT LOGITS
    # ==========================================================

    pruning_parser = subparsers.add_parser(
        "prune",
        help="Generate pruned dataset split."
    )

    pruning_parser.add_argument(
        "--dataset-path",
        required=True
    )

    pruning_parser.add_argument(
        "--pruning-model",
        required=True,
        choices=["medsam", "unet"],
        default="medsam"
    )

    pruning_parser.add_argument(
        "--save-logits",
        required=True,
        default=True
    )

    pruning_parser.add_argument(
        "--pruning-rate",
        required=True,
        type=float,
    )

    pruning_parser.add_argument(
        "--strategy",
        required=True,
        choices=pruning_strategies
    )

    pruning_parser.add_argument(
        "--selection",
        required=True,
        choices=[
            "easy",
            "hard",
            "mixed",
        ],
    )

    pruning_parser.add_argument(
        "--seed",
        default=123,
        type=int,
    )

    pruning_parser.add_argument(
        "--epochs",
        default=10,
        type=int,
    )

    pruning_parser.add_argument(
        "--batch_size",
        default=32,
        type=int
    )

    pruning_parser.add_argument(
        "--lr",
        default=2e-4,
        type=float
    )

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":

        if args.dataset:
            download_dataset(
                dataset_name=args.dataset,
                outdir=args.data_dir,
            )

    elif args.command == "prune":
        pass
        #TODO pipeline pruning

    else:
        raise ValueError(f"Unknown command {args.command}")

    

if __name__ == "__main__":
    main()